import os
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import requests
import pandas as pd

from .base_source import BaseSource


LEAGUES = ["PL", "BL1", "PD", "SA", "FL1", "OSL"]

# football-data.org competition IDs
LEAGUE_IDS = {
    "PL":  2021,  # Premier League
    "BL1": 2002,  # Bundesliga
    "PD":  2014,  # La Liga (Primera Division)
    "SA":  2019,  # Serie A
    "FL1": 2015,  # Ligue 1
    "OSL": 2003,  # Eredivisie / kept for compat; swap to actual ID as needed
}


class FootballAPISource(BaseSource):
    """Client for the football-data.org v4 REST API.

    Rate limit: max 10 requests per 60-second rolling window (free tier).
    """

    BASE_URL = "https://api.football-data.org/v4"
    MAX_REQUESTS_PER_MINUTE = 10

    def __init__(self):
        super().__init__()
        api_key = os.environ.get("FOOTBALL_API_KEY", "")
        if not api_key:
            self.logger.warning("FOOTBALL_API_KEY not set – requests may be rejected.")
        self.session = requests.Session()
        self.session.headers.update({
            "X-Auth-Token": api_key,
            "Content-Type": "application/json",
        })
        # Rate-limiter state
        self._lock = threading.Lock()
        self._request_timestamps: List[float] = []

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        """Block until making a new request stays within the rate limit."""
        with self._lock:
            now = time.monotonic()
            window_start = now - 60.0
            # Remove timestamps outside the 60-second window
            self._request_timestamps = [
                ts for ts in self._request_timestamps if ts > window_start
            ]
            if len(self._request_timestamps) >= self.MAX_REQUESTS_PER_MINUTE:
                oldest = self._request_timestamps[0]
                sleep_for = 60.0 - (now - oldest) + 0.1  # small buffer
                if sleep_for > 0:
                    self.logger.debug(f"Rate limit reached – sleeping {sleep_for:.1f}s")
                    time.sleep(sleep_for)
            self._request_timestamps.append(time.monotonic())

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        self._throttle()
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Public fetch methods
    # ------------------------------------------------------------------

    def fetch_standings(self, league_code: str) -> Dict:
        """Return raw standings payload for *league_code* (e.g. 'PL')."""
        competition_id = LEAGUE_IDS.get(league_code)
        if competition_id is None:
            raise ValueError(f"Unknown league code: {league_code}")
        return self._get(f"competitions/{competition_id}/standings")

    def fetch_matches(
        self,
        league_code: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict:
        """Return raw matches for *league_code*.

        Args:
            league_code: e.g. 'PL'
            date_from:   ISO date string 'YYYY-MM-DD'  (default: today)
            date_to:     ISO date string 'YYYY-MM-DD'  (default: today + 14 days)
        """
        competition_id = LEAGUE_IDS.get(league_code)
        if competition_id is None:
            raise ValueError(f"Unknown league code: {league_code}")

        today = datetime.utcnow().date()
        params: Dict[str, str] = {
            "dateFrom": date_from or str(today),
            "dateTo":   date_to   or str(today + timedelta(days=14)),
        }
        return self._get(f"competitions/{competition_id}/matches", params=params)

    def fetch_head_to_head(self, match_id: int, limit: int = 10) -> Dict:
        """Return head-to-head data for a specific *match_id*."""
        return self._get(f"matches/{match_id}/head2head", params={"limit": limit})

    # ------------------------------------------------------------------
    # BaseSource interface
    # ------------------------------------------------------------------

    def fetch(self) -> Dict:
        """Fetch all matches across all supported leagues for the next 14 days."""
        all_matches: List[Dict] = []
        today = datetime.utcnow().date()
        date_from = str(today)
        date_to   = str(today + timedelta(days=14))

        for league in LEAGUES:
            try:
                data = self.fetch_matches(league, date_from, date_to)
                for match in data.get("matches", []):
                    match["_league_code"] = league
                    all_matches.append(match)
            except Exception as exc:
                self.logger.warning(f"Could not fetch matches for {league}: {exc}")

        return {"matches": all_matches}

    def validate(self, data: Dict) -> bool:
        if not isinstance(data, dict):
            return False
        if "matches" not in data:
            self.logger.warning("Validation failed: 'matches' key missing")
            return False
        return True

    def normalize(self, data: Dict) -> pd.DataFrame:
        """Normalize raw match data to the standard internal schema."""
        rows = []
        for m in data.get("matches", []):
            score  = m.get("score", {})
            full   = score.get("fullTime", {})
            home_goals = full.get("home")
            away_goals = full.get("away")

            utc_date = m.get("utcDate", "")
            try:
                date_parsed = datetime.fromisoformat(utc_date.replace("Z", "+00:00"))
                date_str = date_parsed.strftime("%Y-%m-%d")
            except Exception:
                date_str = utc_date[:10] if utc_date else None

            rows.append({
                "match_id":   m.get("id"),
                "date":       date_str,
                "league":     m.get("_league_code") or m.get("competition", {}).get("code"),
                "home_team":  m.get("homeTeam", {}).get("name"),
                "away_team":  m.get("awayTeam", {}).get("name"),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "status":     m.get("status"),
                "matchday":   m.get("matchday"),
                "season":     m.get("season", {}).get("startDate", "")[:4],
            })

        df = pd.DataFrame(rows, columns=[
            "match_id", "date", "league", "home_team", "away_team",
            "home_goals", "away_goals", "status", "matchday", "season",
        ])
        return df
