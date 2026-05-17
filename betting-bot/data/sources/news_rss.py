"""News RSS sentiment source.

Parses a set of sports RSS feeds and computes a per-match sentiment modifier
based on keyword matches for injury/suspension news (negative) and
return-to-fitness news (positive).

The ``get_match_sentiment`` method returns::

    {
        "snippets": [
            {"feed": "BBC Sport", "title": "...", "summary": "...", "sentiment": -1},
            ...
        ],
        "net_sentiment_modifier": float  # clipped to [-0.10, +0.05]
    }

A net_sentiment_modifier of -0.10 should reduce a model's confidence
significantly, while +0.05 is a small boost.
"""
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, timedelta

try:
    import feedparser
    _FEEDPARSER_AVAILABLE = True
except ImportError:  # Python 3.13+ / missing sgmllib fallback
    _FEEDPARSER_AVAILABLE = False

import pandas as pd

from .base_source import BaseSource


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RSS_FEEDS: List[Dict[str, str]] = [
    {"name": "BBC Sport Football",  "url": "https://feeds.bbci.co.uk/sport/football/rss.xml"},
    {"name": "Sky Sports Football", "url": "https://www.skysports.com/rss/12040"},
    {"name": "ESPN NBA",            "url": "https://www.espn.com/espn/rss/nba/news"},
    {"name": "Kicker Bundesliga",   "url": "https://www.kicker.de/news/fussball/bundesliga/rss.xml"},
]

NEGATIVE_KEYWORDS: List[str] = [
    "verletzt", "fehlt", "fällt aus", "fraglich",
    "injured", "out", "doubt", "suspended", "ban",
    "injury concern", "rotation expected", "rested",
    "geschont", "gesperrt",
]

POSITIVE_KEYWORDS: List[str] = [
    "fit again", "returns", "available", "full squad",
    "zurück", "wieder fit", "vollständig",
]

# Weight per keyword hit (negative = -0.02, positive = +0.01)
NEGATIVE_WEIGHT: float = -0.02
POSITIVE_WEIGHT: float = +0.01

# Clip range for the net sentiment modifier
MIN_MODIFIER: float = -0.10
MAX_MODIFIER: float = +0.05

# How far back to consider news items (in hours)
NEWS_LOOKBACK_HOURS: int = 72


class NewsRSSSource(BaseSource):
    """Fetch and parse sports RSS feeds for pre-match sentiment analysis."""

    def __init__(self, feeds: Optional[List[Dict[str, str]]] = None):
        """
        Args:
            feeds: Override default feed list.  Each dict must have 'name' and 'url'.
        """
        super().__init__()
        self.feeds = feeds or RSS_FEEDS
        # In-process cache: feed URL → list of parsed entries
        self._feed_cache: Dict[str, List[Dict]] = {}
        self._fetch_errors: List[str] = []

    # ------------------------------------------------------------------
    # Feed fetching
    # ------------------------------------------------------------------

    def _fetch_feed(self, feed_url: str) -> List[Dict]:
        """Fetch and return parsed entries for *feed_url*.

        Uses feedparser which handles HTTP itself.  Returns empty list on error
        or if feedparser is unavailable.
        """
        if not _FEEDPARSER_AVAILABLE:
            self.logger.warning("feedparser not available – skipping RSS feeds.")
            return []
        try:
            parsed = feedparser.parse(feed_url)
            if parsed.bozo and not parsed.entries:
                self.logger.warning(
                    f"RSS feed parse warning for {feed_url}: {parsed.bozo_exception}"
                )
            entries = []
            for entry in parsed.entries:
                entries.append({
                    "title":     getattr(entry, "title", "") or "",
                    "summary":   getattr(entry, "summary", "") or "",
                    "link":      getattr(entry, "link", "") or "",
                    "published": getattr(entry, "published", "") or "",
                    "feed_url":  feed_url,
                })
            return entries
        except Exception as exc:
            self.logger.error(f"Failed to fetch RSS feed {feed_url}: {exc}")
            self._fetch_errors.append(f"{feed_url}: {exc}")
            return []

    def _fetch_all_feeds(self) -> Dict[str, List[Dict]]:
        """Fetch all configured feeds and cache them."""
        self._feed_cache = {}
        for feed in self.feeds:
            url = feed["url"]
            self.logger.info(f"Fetching RSS: {feed['name']} ({url})")
            entries = self._fetch_feed(url)
            self._feed_cache[url] = entries
            self.logger.info(f"  → {len(entries)} entries")
        return self._feed_cache

    # ------------------------------------------------------------------
    # Keyword matching
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_score(text: str) -> int:
        """Return raw keyword score for *text* (sum of hits, neg or pos)."""
        text_lower = text.lower()
        score = 0
        for kw in NEGATIVE_KEYWORDS:
            if kw.lower() in text_lower:
                score -= 1
        for kw in POSITIVE_KEYWORDS:
            if kw.lower() in text_lower:
                score += 1
        return score

    @staticmethod
    def _entry_mentions_team(entry: Dict, team_name: str) -> bool:
        """Return True if *entry* title or summary mentions *team_name*."""
        if not team_name:
            return False
        needle = team_name.lower()
        combined = (entry.get("title", "") + " " + entry.get("summary", "")).lower()
        return needle in combined

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_match_sentiment(
        self,
        home_team: str,
        away_team: str,
        refresh_feeds: bool = True,
    ) -> Dict[str, Any]:
        """Compute news sentiment modifier for a match.

        Args:
            home_team:     Home team name (as it appears in news headlines).
            away_team:     Away team name.
            refresh_feeds: If True, re-fetch all feeds before analysis.

        Returns:
            {
                "snippets": [
                    {
                        "feed_url": str,
                        "title": str,
                        "summary": str,
                        "sentiment": int,   # raw score: neg < 0, pos > 0
                    },
                    ...
                ],
                "net_sentiment_modifier": float   # clipped to [-0.10, +0.05]
            }
        """
        if refresh_feeds or not self._feed_cache:
            self._fetch_all_feeds()

        snippets: List[Dict] = []
        total_score = 0

        all_entries: List[Dict] = []
        for entries in self._feed_cache.values():
            all_entries.extend(entries)

        for entry in all_entries:
            # Filter to entries that mention either team
            if not (
                self._entry_mentions_team(entry, home_team)
                or self._entry_mentions_team(entry, away_team)
            ):
                continue

            combined_text = entry.get("title", "") + " " + entry.get("summary", "")
            score = self._keyword_score(combined_text)
            if score == 0:
                continue  # no relevant keywords – skip

            total_score += score
            snippets.append({
                "feed_url":  entry.get("feed_url", ""),
                "title":     entry.get("title", ""),
                "summary":   entry.get("summary", "")[:300],
                "sentiment": score,
            })

        # Convert total_score to a modifier and clip
        if total_score < 0:
            modifier = max(total_score * abs(NEGATIVE_WEIGHT), MIN_MODIFIER)
        else:
            modifier = min(total_score * POSITIVE_WEIGHT, MAX_MODIFIER)

        return {
            "snippets":              snippets,
            "net_sentiment_modifier": round(modifier, 4),
        }

    def get_all_recent_entries(self) -> List[Dict]:
        """Return all cached feed entries (fetching first if cache is empty)."""
        if not self._feed_cache:
            self._fetch_all_feeds()
        all_entries: List[Dict] = []
        for entries in self._feed_cache.values():
            all_entries.extend(entries)
        return all_entries

    # ------------------------------------------------------------------
    # BaseSource interface
    # ------------------------------------------------------------------

    def fetch(self) -> Dict:
        """Fetch all RSS feeds and return raw entry data."""
        feed_data = self._fetch_all_feeds()
        all_entries: List[Dict] = []
        for url, entries in feed_data.items():
            feed_name = next(
                (f["name"] for f in self.feeds if f["url"] == url), url
            )
            for e in entries:
                all_entries.append({**e, "feed_name": feed_name})

        return {
            "entries": all_entries,
            "errors":  list(self._fetch_errors),
        }

    def validate(self, data: Dict) -> bool:
        if not isinstance(data, dict) or "entries" not in data:
            return False
        return True

    def normalize(self, data: Dict) -> pd.DataFrame:
        """Return a DataFrame of all fetched news entries with sentiment scores."""
        entries = data.get("entries", [])
        rows = []
        for entry in entries:
            combined = entry.get("title", "") + " " + entry.get("summary", "")
            score = self._keyword_score(combined)
            rows.append({
                "feed_name": entry.get("feed_name", ""),
                "feed_url":  entry.get("feed_url", ""),
                "title":     entry.get("title", ""),
                "summary":   (entry.get("summary") or "")[:300],
                "link":      entry.get("link", ""),
                "published": entry.get("published", ""),
                "sentiment_score": score,
            })

        expected_cols = [
            "feed_name", "feed_url", "title", "summary",
            "link", "published", "sentiment_score",
        ]
        if not rows:
            return pd.DataFrame(columns=expected_cols)
        df = pd.DataFrame(rows)
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None
        return df[expected_cols]
