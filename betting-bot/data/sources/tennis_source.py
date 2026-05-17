"""Tennis data source for WTA and ITF women's/men's events.

Data is sourced from:
- Tennis Abstract (Jeff Sackmann GitHub CSVs): historical match results
  https://github.com/JeffSackmann/tennis_wta  (WTA)
  https://github.com/JeffSackmann/tennis_atp  (ATP/ITF men)
- The-Odds-API: live odds for WTA events (h2h market)
  https://api.the-odds-api.com/v4/sports/tennis/odds/

DOCUMENTED LIMITATION:
  The-Odds-API covers WTA tour events but does NOT cover ITF M15/F10
  challengers.  fetch_odds() returns None for those markets and callers
  must handle the None case gracefully.

CSV caching strategy:
  CSVs are cached under data/cache/tennis/.
  A file is re-downloaded only when it is older than 24 hours (or absent).
  Current-year files change as the season progresses; prior-year files are
  effectively static but are still subject to the same 24-hour rule.
"""

import csv
import io
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from data.sources.base_source import BaseSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS = 86_400  # 24 hours

# Jeff Sackmann CSV URLs
_WTA_URL_TEMPLATE  = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/"
    "wta_matches_{year}.csv"
)
_ATP_URL_TEMPLATE  = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"
    "atp_matches_{year}.csv"
)

# Tourney level codes present in the Sackmann CSVs.
# 'G' = Grand Slam, 'F' = Finals, 'PM'/'M' = Premier/Masters,
# 'A' = standard WTA tour, 'C' = Challenger, 'S' = ITF/satellite.
# We target the mid-tier where market inefficiencies are most likely.
_TARGET_TOURNEY_LEVELS = {"A", "C", "S", "125", "ITF"}

# Ranking range of interest (inclusive).  Matches outside this range have
# either too much efficient-market pricing (top 50) or too little data.
_MIN_RANKING = 50
_MAX_RANKING = 300

# The-Odds-API endpoint
_ODDS_API_BASE = "https://api.the-odds-api.com/v4"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _cache_dir() -> Path:
    """Return the tennis cache directory, creating it if necessary."""
    # Resolve relative to this file so the path works regardless of cwd.
    here = Path(__file__).resolve().parent
    # Go up two levels: sources/ -> data/ -> betting-bot/data/cache/tennis/
    cache = here.parent / "cache" / "tennis"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _is_stale(path: Path) -> bool:
    """Return True if path does not exist or is older than 24 hours."""
    if not path.exists():
        return True
    age = time.time() - path.stat().st_mtime
    return age >= _CACHE_TTL_SECONDS


def _download_csv(url: str, dest: Path, timeout: int = 15) -> bool:
    """Download *url* to *dest*.  Returns True on success."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "betting-bot/1.0"})
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        logger.debug(f"TennisSource: downloaded {url} → {dest}")
        return True
    except Exception as exc:
        logger.warning(f"TennisSource: failed to download {url}: {exc}")
        return False


def _load_csv_cached(url: str, filename: str, timeout: int = 15) -> Optional[pd.DataFrame]:
    """Return DataFrame from cache, refreshing if stale."""
    dest = _cache_dir() / filename
    if _is_stale(dest):
        ok = _download_csv(url, dest, timeout=timeout)
        if not ok and not dest.exists():
            return None

    try:
        df = pd.read_csv(dest, low_memory=False)
        return df
    except Exception as exc:
        logger.warning(f"TennisSource: failed to parse {dest}: {exc}")
        return None


# ---------------------------------------------------------------------------
# TennisSource
# ---------------------------------------------------------------------------

class TennisSource(BaseSource):
    """Fetch and normalize tennis match data from Sackmann CSVs and Odds API.

    Compatible with the :class:`BaseSource` abstract interface.
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        self.config  = config or {}
        ds           = self.config.get("data_sources", {})
        self.api_key  = ds.get("odds_api_key", os.environ.get("ODDS_API_KEY", ""))
        self.timeout  = ds.get("timeout_seconds", 15)
        self._current_year  = datetime.now(timezone.utc).year
        self._previous_year = self._current_year - 1

    # ------------------------------------------------------------------
    # BaseSource interface — fetch / validate / normalize
    # ------------------------------------------------------------------

    def fetch(self) -> dict:
        """Download WTA and ATP/ITF CSVs for the current and previous year.

        Returns a dict keyed by tour ('wta', 'atp') with sub-keys per year.
        Each value is a :class:`pandas.DataFrame` or None if unavailable.
        """
        data: dict = {"wta": {}, "atp": {}}

        for year in (self._current_year, self._previous_year):
            # WTA
            wta_url  = _WTA_URL_TEMPLATE.format(year=year)
            wta_file = f"wta_matches_{year}.csv"
            data["wta"][year] = _load_csv_cached(wta_url, wta_file, self.timeout)

            # ATP / ITF men
            atp_url  = _ATP_URL_TEMPLATE.format(year=year)
            atp_file = f"atp_matches_{year}.csv"
            data["atp"][year] = _load_csv_cached(atp_url, atp_file, self.timeout)

        return data

    def validate(self, data: dict) -> bool:
        """Return True if at least one DataFrame was loaded successfully."""
        if not isinstance(data, dict):
            return False
        for tour_data in data.values():
            for df in tour_data.values():
                if df is not None and not df.empty:
                    return True
        return False

    def normalize(self, data: dict) -> pd.DataFrame:
        """Convert raw Sackmann CSV DataFrames to the standard match format.

        Applies tourney-level and ranking filters.  Only rows where both
        players have rankings in [_MIN_RANKING, _MAX_RANKING] are kept.

        Returns a DataFrame with columns::

            match_id, date, sport, home_team, away_team, league,
            surface, player1_ranking, player2_ranking, tourney_level, status
        """
        frames: List[pd.DataFrame] = []

        for tour, year_map in data.items():
            for year, df in year_map.items():
                if df is None or df.empty:
                    continue
                normalized = self._normalize_sackmann_df(df, tour=tour)
                if not normalized.empty:
                    frames.append(normalized)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["match_id"])
        return combined

    # ------------------------------------------------------------------
    # Primary domain methods
    # ------------------------------------------------------------------

    # Tennis sport keys available on the-odds-api for upcoming fixtures
    _ODDS_API_TENNIS_SPORT_KEYS = ["tennis_wta"]

    def fetch_upcoming_matches(self, days_ahead: int = 3) -> List[Dict]:
        """Return upcoming WTA tennis matches within the next *days_ahead* days.

        Uses the-odds-api /v4/sports/{sport_key}/events endpoint to obtain
        real future fixtures.  Sackmann CSVs only contain completed results
        and cannot reliably supply upcoming fixtures.

        Args:
            days_ahead: How many calendar days into the future to include.

        Returns:
            List of dicts with keys:
                match_id, date, sport, home_team (player1),
                away_team (player2), league (sport_key), surface
            surface is always None — the events endpoint does not supply it.
        """
        if not self.api_key:
            logger.warning(
                "TennisSource.fetch_upcoming_matches: no ODDS_API_KEY configured — "
                "cannot fetch upcoming fixtures from the-odds-api"
            )
            return []

        now      = datetime.now(timezone.utc)
        cutoff   = now + timedelta(days=days_ahead)
        results: List[Dict] = []

        for sport_key in self._ODDS_API_TENNIS_SPORT_KEYS:
            url = f"{_ODDS_API_BASE}/sports/{sport_key}/events"
            try:
                resp = requests.get(
                    url,
                    params={"apiKey": self.api_key, "dateFormat": "iso"},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                events = resp.json()
            except Exception as exc:
                logger.error(
                    f"TennisSource.fetch_upcoming_matches: API call failed for "
                    f"{sport_key}: {exc}"
                )
                continue

            for event in events:
                commence_raw = event.get("commence_time", "")
                try:
                    commence_dt = datetime.fromisoformat(
                        commence_raw.replace("Z", "+00:00")
                    )
                except Exception:
                    logger.debug(
                        f"TennisSource: could not parse commence_time "
                        f"'{commence_raw}' for event {event.get('id')}"
                    )
                    continue

                # Keep only events within the look-ahead window
                if not (now <= commence_dt <= cutoff):
                    continue

                results.append({
                    "match_id":   event.get("id", ""),
                    "date":       commence_dt.strftime("%Y-%m-%d"),
                    "sport":      "tennis",
                    "home_team":  event.get("home_team", ""),
                    "away_team":  event.get("away_team", ""),
                    "league":     event.get("sport_key", sport_key),
                    "surface":    None,  # events endpoint does not supply surface
                })

        logger.info(
            f"TennisSource: {len(results)} upcoming WTA matches in next {days_ahead} days"
        )
        return results

    def fetch_itf_upcoming(self) -> List[Dict]:
        """Return upcoming ITF M15/F10 fixtures.

        ITF M15 and F10 challenger events have no free coverage on
        the-odds-api.  This method returns an empty list and logs the
        limitation explicitly so callers are never silently left without
        data due to a missing code path.

        Returns:
            Empty list — ITF coverage is unavailable via free tier APIs.
        """
        logger.info(
            "TennisSource.fetch_itf_upcoming: ITF M15/F10 challenger events have no "
            "free odds-API coverage — returning empty list.  To obtain ITF fixtures "
            "a paid data provider (e.g. SportRadar, API-Tennis) is required."
        )
        return []

    def fetch_odds(self, player1: str, player2: str) -> Optional[Dict]:
        """Fetch h2h odds for a match from The-Odds-API.

        LIMITATION: The-Odds-API covers WTA tour events but NOT ITF M15/F10.
        Returns None if no match is found or if the API call fails.

        Args:
            player1: Name of the first player (home_team convention).
            player2: Name of the second player (away_team convention).

        Returns:
            Dict with keys::

                player1, player2,
                player1_odds, player2_odds,  (best available decimal odds)
                bookmaker, market

            or None if odds are unavailable.
        """
        if not self.api_key:
            logger.debug("TennisSource.fetch_odds: no ODDS_API_KEY configured")
            return None

        try:
            resp = requests.get(
                f"{_ODDS_API_BASE}/sports/tennis/odds/",
                params={
                    "apiKey":  self.api_key,
                    "markets": "h2h",
                    "regions": "eu,uk,us",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            events = resp.json()
        except Exception as exc:
            logger.warning(f"TennisSource.fetch_odds: API call failed: {exc}")
            return None

        # Search for the match involving player1 and player2 (case-insensitive,
        # partial match to handle name-format differences).
        p1_lower = player1.lower()
        p2_lower = player2.lower()

        for event in events:
            home = event.get("home_team", "").lower()
            away = event.get("away_team", "").lower()

            # Check both orderings
            if not (
                (p1_lower in home or home in p1_lower) and
                (p2_lower in away or away in p2_lower)
            ) and not (
                (p2_lower in home or home in p2_lower) and
                (p1_lower in away or away in p1_lower)
            ):
                continue

            # Found the match — extract best odds across all bookmakers
            best_p1_odds: Optional[float] = None
            best_p2_odds: Optional[float] = None
            best_bookmaker = ""

            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    outcomes = market.get("outcomes", [])
                    p1_outcome = next(
                        (o for o in outcomes if p1_lower in o.get("name", "").lower()),
                        None,
                    )
                    p2_outcome = next(
                        (o for o in outcomes if p2_lower in o.get("name", "").lower()),
                        None,
                    )
                    if p1_outcome and p2_outcome:
                        p1_price = p1_outcome.get("price", 0.0)
                        p2_price = p2_outcome.get("price", 0.0)
                        if best_p1_odds is None or p1_price > best_p1_odds:
                            best_p1_odds   = p1_price
                            best_p2_odds   = p2_price
                            best_bookmaker = bookmaker.get("key", "")

            if best_p1_odds is not None:
                return {
                    "player1":       player1,
                    "player2":       player2,
                    "player1_odds":  best_p1_odds,
                    "player2_odds":  best_p2_odds,
                    "bookmaker":     best_bookmaker,
                    "market":        "h2h",
                }

        logger.debug(
            f"TennisSource.fetch_odds: no odds found for {player1} vs {player2} "
            "(ITF events are not covered by The-Odds-API)"
        )
        return None

    def fetch_results(self, match_date: date) -> List[Dict]:
        """Return finished matches from Sackmann CSVs for *match_date*.

        Args:
            match_date: The date to look up (``datetime.date`` object).

        Returns:
            List of normalized match dicts for that date where the match is
            marked as completed (i.e., a winner is recorded in the CSV).
        """
        raw = self.fetch()
        if not self.validate(raw):
            return []

        combined = self.normalize(raw)
        if combined.empty:
            return []

        combined["_date_parsed"] = pd.to_datetime(combined["date"], errors="coerce").dt.date
        on_date = combined[combined["_date_parsed"] == match_date].copy()
        on_date = on_date.drop(columns=["_date_parsed"], errors="ignore")

        # In Sackmann CSVs a row always represents a *completed* match
        # (the winner is always recorded). Filter to "finished" status.
        finished = on_date[on_date.get("status", pd.Series("finished", index=on_date.index)) == "finished"]
        if "status" not in on_date.columns:
            finished = on_date  # all rows are results

        return finished.to_dict(orient="records")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_sackmann_df(self, df: pd.DataFrame, tour: str) -> pd.DataFrame:
        """Convert a raw Sackmann CSV DataFrame into the standard format.

        The Sackmann CSVs use these relevant columns (names may vary slightly):
            tourney_id, tourney_name, surface, tourney_level, tourney_date,
            match_num, winner_name, winner_rank, loser_name, loser_rank

        tourney_date is YYYYMMDD.  match_num is a 3-digit integer within the
        tournament.  A composite match_id is built as
        ``{tourney_id}_{match_num}``.
        """
        df = df.copy()

        # Normalise column names to lowercase with underscores
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        required = {"tourney_name", "surface", "tourney_level", "tourney_date",
                    "winner_name", "loser_name"}
        missing = required - set(df.columns)
        if missing:
            logger.debug(f"TennisSource: missing columns {missing} in {tour} CSV — skipping")
            return pd.DataFrame()

        # Parse date
        df["_date"] = pd.to_datetime(
            df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
        ).dt.date

        # Tourney level filter
        df["_level_upper"] = df["tourney_level"].astype(str).str.upper()
        df = df[df["_level_upper"].isin({lvl.upper() for lvl in _TARGET_TOURNEY_LEVELS})]

        # Ranking columns — handle missing gracefully
        for col in ("winner_rank", "loser_rank"):
            if col not in df.columns:
                df[col] = None

        df["_winner_rank"] = pd.to_numeric(df["winner_rank"], errors="coerce")
        df["_loser_rank"]  = pd.to_numeric(df["loser_rank"],  errors="coerce")

        # Ranking filter (both players must be in range; NaN is treated as out-of-range)
        rank_mask = (
            df["_winner_rank"].between(_MIN_RANKING, _MAX_RANKING) &
            df["_loser_rank"].between(_MIN_RANKING, _MAX_RANKING)
        )
        df = df[rank_mask]

        if df.empty:
            return pd.DataFrame()

        # Build composite match_id
        tourney_id_col = "tourney_id" if "tourney_id" in df.columns else None
        match_num_col  = "match_num"  if "match_num"  in df.columns else None

        if tourney_id_col and match_num_col:
            df["match_id"] = (
                df[tourney_id_col].astype(str) + "_" + df[match_num_col].astype(str)
            )
        else:
            # Fallback: hash the key fields
            df["match_id"] = (
                df["_date"].astype(str)
                + "_" + df["winner_name"].astype(str)
                + "_" + df["loser_name"].astype(str)
            )

        rows = []
        for _, row in df.iterrows():
            # Randomize which player is "home" (player1) using a deterministic coin
            # flip based on match_id hash.  This eliminates winner-label leakage:
            # XGBoost would otherwise learn "player1 always wins" from Sackmann CSVs
            # where winner is unconditionally stored as the first player.
            import hashlib
            flip = int(hashlib.md5(str(row["match_id"]).encode()).hexdigest(), 16) % 2 == 0
            if flip:
                p1_name, p2_name = str(row["winner_name"]), str(row["loser_name"])
                p1_rank, p2_rank = row["_winner_rank"], row["_loser_rank"]
                p1_is_winner = True
            else:
                p1_name, p2_name = str(row["loser_name"]), str(row["winner_name"])
                p1_rank, p2_rank = row["_loser_rank"], row["_winner_rank"]
                p1_is_winner = False

            rows.append({
                "match_id":          str(row["match_id"]),
                "date":              str(row["_date"]) if pd.notna(row["_date"]) else "",
                "sport":             "tennis",
                "home_team":         p1_name,
                "away_team":         p2_name,
                "league":            str(row["tourney_name"]),
                "surface":           str(row.get("surface", "")),
                "player1_ranking":   p1_rank if pd.notna(p1_rank) else None,
                "player2_ranking":   p2_rank if pd.notna(p2_rank) else None,
                "tourney_level":     str(row["tourney_level"]),
                "status":            "finished",
                "tour":              tour,
                # Training target: was player1 the historical winner?
                # XGBoost must predict this — not "player1 wins" by construction.
                "player1_is_winner": p1_is_winner,
            })

        return pd.DataFrame(rows)
