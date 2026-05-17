"""Hockey data source for AHL, ECHL, and European minor leagues.

Data is sourced from:
- EliteProspects API (free tier): team schedules, rosters, results
  https://www.eliteprospects.com/api
- SofaScore scraping (fallback): live scores and goalie lineups
  https://www.sofascore.com
- Flashscore scraping (backup): results

Key hockey-specific features:
- Back-to-back flag: Did the team play yesterday?
- Days rest: How many days since the last game?
- B2B penalty: Performance multiplier when playing back-to-back
- Goalie confirmation: Is the starting goalie known? (handled via late-news hook)

IMPORTANT — Goalie risk:
  European minor league goalie lineups are often confirmed only 60-90 minutes
  before puck drop.  The LateNewsHook in the scheduler must re-check SofaScore
  and abort/flag any bet where the confirmed goalie differs from expected.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from data.sources.base_source import BaseSource

logger = logging.getLogger(__name__)

# Performance penalty applied to expected goals / win probability when a team
# is playing back-to-back (second game in two days).
# Source: Academic literature on NHL B2B — ~3-5% win rate drop.
# Applied conservatively at 4% for minor leagues.
B2B_WIN_RATE_PENALTY = 0.04


class HockeySource(BaseSource):
    """Fetch and normalize hockey data from EliteProspects and SofaScore."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.api_key = (
            (config or {})
            .get("data_sources", {})
            .get("eliteprospects_api_key", "")
        )
        self.base_url = "https://api.eliteprospects.com/v1"
        self.timeout  = (config or {}).get("data_sources", {}).get("timeout_seconds", 10)
        self._session_cache: dict = {}

    # ------------------------------------------------------------------
    # BaseSource interface
    # ------------------------------------------------------------------

    def fetch(self) -> dict:
        """Fetch upcoming games and recent results."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = requests.get(
                f"{self.base_url}/games",
                params={"status": "upcoming", "limit": 100},
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"HockeySource.fetch() failed: {e}")
            return {}

    def validate(self, data: dict) -> bool:
        return isinstance(data, dict) and "data" in data

    def normalize(self, data: dict) -> pd.DataFrame:
        """Convert raw EliteProspects response to standard match format."""
        rows = []
        for game in data.get("data", []):
            try:
                rows.append({
                    "match_id":   str(game.get("id", "")),
                    "date":       game.get("startAt", "")[:10],
                    "league":     game.get("league", {}).get("slug", ""),
                    "home_team":  game.get("homeTeam", {}).get("name", ""),
                    "away_team":  game.get("awayTeam", {}).get("name", ""),
                    "home_goals": game.get("homeScore"),
                    "away_goals": game.get("awayScore"),
                    "status":     game.get("status", ""),
                    "sport":      "hockey",
                })
            except Exception as e:
                logger.debug(f"Hockey game parse error: {e}")
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Back-to-back detection
    # ------------------------------------------------------------------

    def get_back_to_back_flag(
        self,
        team: str,
        game_date: str,
        matches_df: Optional[pd.DataFrame],
    ) -> dict:
        """Determine if a team is playing back-to-back on game_date.

        Args:
            team:       Team name as it appears in matches_df.
            game_date:  ISO date string of the upcoming game (YYYY-MM-DD).
            matches_df: DataFrame of completed matches.  Must have columns:
                        date, home_team, away_team, status.

        Returns:
            Dict with keys:
            - back_to_back (bool): True if team played yesterday.
            - days_since_last_game (int): Calendar days since last game.
              Returns 14 if no prior game found (assume rested).
            - b2b_penalty (float): Win probability adjustment to apply.
              Negative value (e.g. -0.04) if B2B, else 0.0.
        """
        default = {"back_to_back": False, "days_since_last_game": 14, "b2b_penalty": 0.0}

        if matches_df is None or matches_df.empty:
            return default

        try:
            target_date = datetime.strptime(game_date[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            logger.warning(f"HockeySource: invalid game_date '{game_date}'")
            return default

        df = matches_df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Filter to completed matches involving this team
        finished_statuses = {"finished", "ft", "complete", "final", ""}
        if "status" in df.columns:
            df = df[df["status"].str.lower().isin(finished_statuses)]

        team_games = df[
            (df.get("home_team", pd.Series(dtype=str)) == team) |
            (df.get("away_team", pd.Series(dtype=str)) == team)
        ]

        if team_games.empty:
            return default

        if "date" not in team_games.columns:
            return default

        team_games = team_games.copy()
        team_games["_date"] = pd.to_datetime(team_games["date"], errors="coerce").dt.date
        team_games = team_games.dropna(subset=["_date"])
        team_games = team_games[team_games["_date"] < target_date]

        if team_games.empty:
            return default

        last_game_date = team_games["_date"].max()
        days_rest = (target_date - last_game_date).days

        is_b2b = days_rest == 1

        return {
            "back_to_back":        is_b2b,
            "days_since_last_game": int(days_rest),
            "b2b_penalty":         -B2B_WIN_RATE_PENALTY if is_b2b else 0.0,
        }

    # ------------------------------------------------------------------
    # Goalie confirmation (called by late-news hook)
    # ------------------------------------------------------------------

    def get_goalie_status(self, team: str, match_id: str) -> dict:
        """Attempt to fetch confirmed starting goalie from SofaScore.

        This is called by the LateNewsHook roughly 90 minutes before puck
        drop.  If the confirmed goalie differs from what was expected at
        prediction time, the bet should be flagged for manual review.

        Returns:
            Dict with:
            - confirmed (bool): Whether a starter is confirmed.
            - goalie_name (str): Name if confirmed, '' otherwise.
            - source (str): Which source confirmed it.
        """
        # Attempt SofaScore scraping (best coverage for minor leagues)
        sofascore_url = (
            self.config.get("data_sources", {})
            .get("sofascore_base_url", "https://www.sofascore.com")
        )
        try:
            # SofaScore uses numeric event IDs — this requires match_id mapping.
            # If not available, fall back to "unconfirmed".
            resp = requests.get(
                f"{sofascore_url}/api/v1/event/{match_id}/lineups",
                timeout=self.timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            data = resp.json()
            home_lineup = data.get("home", {}).get("players", [])
            for player in home_lineup:
                if player.get("position", "") == "G" and player.get("substitute") is False:
                    return {"confirmed": True, "goalie_name": player.get("name", ""), "source": "sofascore"}
        except Exception as e:
            logger.debug(f"SofaScore goalie lookup failed for {team}: {e}")

        return {"confirmed": False, "goalie_name": "", "source": "none"}

    # ------------------------------------------------------------------
    # Team schedule helper
    # ------------------------------------------------------------------

    def get_team_schedule(self, team: str, season: str = "") -> pd.DataFrame:
        """Fetch full season schedule for a team from EliteProspects."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = requests.get(
                f"{self.base_url}/teams/{team}/games",
                params={"season": season} if season else {},
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return self.normalize(resp.json())
        except Exception as e:
            logger.warning(f"HockeySource.get_team_schedule() failed for {team}: {e}")
            return pd.DataFrame()
