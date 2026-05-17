import json
import random
import re
import time
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
import pandas as pd

from .base_source import BaseSource


# Understat league slug → internal name
LEAGUE_MAP = {
    "EPL":        "EPL",
    "Bundesliga": "Bundesliga",
    "La_liga":    "La Liga",
    "Serie_A":    "Serie A",
    "Ligue_1":    "Ligue 1",
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]

REQUEST_DELAY = 2  # seconds between requests


class UnderstatSource(BaseSource):
    """Scraper for understat.com xG / npxG data.

    Parses the embedded JSON blobs inside ``<script>`` tags on league pages.
    """

    BASE_URL = "https://understat.com/league"

    def __init__(self, season: Optional[str] = None):
        super().__init__()
        from datetime import datetime
        # Default season: current calendar year (understat uses start year of season)
        self.season = season or str(datetime.utcnow().year)
        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_user_agent(self) -> str:
        return random.choice(USER_AGENTS)

    def _get_page(self, url: str) -> str:
        """Fetch HTML with UA rotation and polite delay."""
        self.session.headers.update({"User-Agent": self._random_user_agent()})
        resp = self.session.get(url, timeout=20)
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY)
        return resp.text

    def _extract_json_from_script(self, html: str, variable_name: str) -> Optional[Any]:  # noqa: E501
        """Extract the JSON value assigned to *variable_name* in a <script> tag.

        Understat embeds data like:
            var datesData = JSON.parse('...')
        where the inner string is percent-encoded JSON.
        """
        # Pattern: var <name> = JSON.parse('<encoded_json>')
        pattern = re.compile(
            rf"var\s+{re.escape(variable_name)}\s*=\s*JSON\.parse\('(.+?)'\)",
            re.DOTALL,
        )
        match = pattern.search(html)
        if not match:
            return None

        raw = match.group(1)
        # Understat uses unicode escapes like \x22 (hex) – decode them
        try:
            decoded = raw.encode("utf-8").decode("unicode_escape")
        except Exception:
            decoded = raw

        # Sometimes the string uses \\\" – normalise
        decoded = decoded.replace('\\"', '"')
        try:
            return json.loads(decoded)
        except json.JSONDecodeError:
            # Fallback: try to use requests' percent-unquote via urllib
            from urllib.parse import unquote
            try:
                return json.loads(unquote(raw))
            except Exception:
                return None

    def _parse_matches(self, json_data: Any, league_name: str) -> List[Dict]:
        """Convert understat datesData JSON array to a list of row dicts."""
        if not isinstance(json_data, list):
            return []

        rows = []
        for match in json_data:
            if not isinstance(match, dict):
                continue

            # Understat stores finished matches with xG; skip incomplete
            xg_home_str  = match.get("xG",   {}).get("h", None)
            xg_away_str  = match.get("xG",   {}).get("a", None)
            npxg_home_str = match.get("npxG", {}).get("h", None)
            npxg_away_str = match.get("npxG", {}).get("a", None)

            def _float_or_none(v):
                try:
                    return float(v) if v not in (None, "", "None") else None
                except (TypeError, ValueError):
                    return None

            rows.append({
                "match_id":   match.get("id"),
                "date":       match.get("datetime", "")[:10],
                "league":     league_name,
                "home_team":  match.get("h", {}).get("title"),
                "away_team":  match.get("a", {}).get("title"),
                "xg_home":    _float_or_none(xg_home_str),
                "xg_away":    _float_or_none(xg_away_str),
                "npxg_home":  _float_or_none(npxg_home_str),
                "npxg_away":  _float_or_none(npxg_away_str),
            })
        return rows

    # ------------------------------------------------------------------
    # Public method
    # ------------------------------------------------------------------

    def fetch_league(self, league_slug: str, season: Optional[str] = None) -> List[Dict]:
        """Fetch and parse xG data for *league_slug* (e.g. 'EPL').

        Args:
            league_slug: One of the keys in LEAGUE_MAP.
            season:      Season start year, e.g. '2023'.

        Returns:
            List of row dicts with xG fields.
        """
        season = season or self.season
        url = f"{self.BASE_URL}/{league_slug}/{season}"
        self.logger.info(f"Fetching understat: {url}")
        html = self._get_page(url)

        json_data = self._extract_json_from_script(html, "datesData")
        if json_data is None:
            self.logger.warning(f"Could not extract datesData from {url}")
            return []

        league_name = LEAGUE_MAP.get(league_slug, league_slug)
        return self._parse_matches(json_data, league_name)

    # ------------------------------------------------------------------
    # BaseSource interface
    # ------------------------------------------------------------------

    def fetch(self) -> Dict:
        """Fetch xG data for all configured leagues."""
        all_rows: List[Dict] = []
        errors: List[str] = []

        for slug in LEAGUE_MAP:
            try:
                rows = self.fetch_league(slug, self.season)
                all_rows.extend(rows)
                self.logger.info(f"Understat {slug}: {len(rows)} matches")
            except Exception as exc:
                msg = f"{slug}: {exc}"
                errors.append(msg)
                self.logger.warning(f"Understat failed for {msg}")

        return {"matches": all_rows, "errors": errors}

    def validate(self, data: Dict) -> bool:
        if not isinstance(data, dict) or "matches" not in data:
            return False
        return True

    def normalize(self, data: Dict) -> pd.DataFrame:
        """Return DataFrame with xG columns in the standard schema."""
        rows = data.get("matches", [])
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
            "match_id", "date", "league", "home_team", "away_team",
            "xg_home", "xg_away", "npxg_home", "npxg_away",
        ])
        # Ensure correct column order
        expected = ["match_id", "date", "league", "home_team", "away_team",
                    "xg_home", "xg_away", "npxg_home", "npxg_away"]
        for col in expected:
            if col not in df.columns:
                df[col] = None
        return df[expected]


