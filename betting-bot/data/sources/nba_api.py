"""NBA data source via balldontlie.io v1 API.

Fetches team stats, game schedule, and injury reports, then computes
per-team derived features:
    - days_rest_since_last_game
    - back_to_back  (played yesterday?)
    - home_away_record_last_10
    - offensive_rating, defensive_rating, pace  (rolling 10-game averages)

Environment variable:
    NBA_API_KEY – balldontlie API key (optional; free tier works without one).
"""
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd

from .base_source import BaseSource


class NBASource(BaseSource):
    """Client for the balldontlie.io v1 REST API.

    The free tier is rate-limited to ~60 requests/minute but does not
    require an API key for most endpoints.  Providing NBA_API_KEY unlocks
    higher limits and the injury report endpoint.
    """

    BASE_URL = "https://api.balldontlie.io/v1"
    # How many past games to consider for rolling stats
    ROLLING_WINDOW = 10

    def __init__(self, season: Optional[int] = None):
        """
        Args:
            season: NBA season start year, e.g. 2023 for the 2023-24 season.
                    Defaults to the current or most-recent season.
        """
        super().__init__()
        self.api_key = os.environ.get("NBA_API_KEY", "")
        now = datetime.now(timezone.utc)
        # NBA season starts in October; roll back to previous year if before Oct
        self.season = season or (now.year if now.month >= 10 else now.year - 1)
        self.session = requests.Session()
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = self.api_key
        self.session.headers.update(headers)

    # ------------------------------------------------------------------
    # Low-level HTTP (with pagination)
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        merged: Dict[str, Any] = params.copy() if params else {}
        resp = self.session.get(url, params=merged, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _get_all_pages(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """Auto-paginate through all result pages and return merged data list."""
        params = params.copy() if params else {}
        params.setdefault("per_page", 100)
        all_data: List[Dict] = []
        cursor: Optional[int] = None

        while True:
            if cursor is not None:
                params["cursor"] = cursor
            resp = self._get(endpoint, params)
            data = resp.get("data", [])
            all_data.extend(data)

            meta = resp.get("meta", {})
            next_cursor = meta.get("next_cursor")
            if not next_cursor or not data:
                break
            cursor = next_cursor

        return all_data

    # ------------------------------------------------------------------
    # Public fetch methods
    # ------------------------------------------------------------------

    def fetch_teams(self) -> List[Dict]:
        """Return all NBA teams."""
        return self._get_all_pages("teams")

    def fetch_games(
        self,
        season: Optional[int] = None,
        team_ids: Optional[List[int]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[Dict]:
        """Fetch game records, optionally filtered by season, teams, or date range.

        Args:
            season:    Season start year (defaults to self.season).
            team_ids:  List of team IDs to filter by.
            date_from: ISO date 'YYYY-MM-DD'.
            date_to:   ISO date 'YYYY-MM-DD'.

        Returns:
            List of game dicts as returned by the API.
        """
        params: Dict[str, Any] = {"seasons[]": season or self.season}
        if team_ids:
            params["team_ids[]"] = team_ids
        if date_from:
            params["start_date"] = date_from
        if date_to:
            params["end_date"] = date_to
        return self._get_all_pages("games", params)

    def fetch_player_injuries(self) -> List[Dict]:
        """Fetch the current injury report.

        Note: requires a paid API key on balldontlie.  Returns empty list
        if the endpoint is unavailable or the key is missing.
        """
        if not self.api_key:
            self.logger.info("NBA_API_KEY not set – skipping injury report.")
            return []
        try:
            return self._get_all_pages("player_injuries")
        except requests.HTTPError as exc:
            self.logger.warning(f"Injury endpoint unavailable: {exc}")
            return []

    def fetch_team_season_averages(self, team_id: int) -> Optional[Dict]:
        """Return season average stats for *team_id* (if available)."""
        try:
            data = self._get(
                "season_averages",
                params={"season": self.season, "team_ids[]": team_id},
            )
            averages = data.get("data", [])
            return averages[0] if averages else None
        except Exception as exc:
            self.logger.warning(f"Season averages fetch failed for team {team_id}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute_team_features(
        self, games: List[Dict]
    ) -> Dict[int, Dict[str, Any]]:
        """Compute derived features per team from a list of completed games.

        Returns:
            Dict keyed by team_id with features:
                days_rest_since_last_game, back_to_back,
                home_record_last_10, away_record_last_10,
                offensive_rating, defensive_rating, pace
        """
        # Group completed games by team, sorted by date
        team_games: Dict[int, List[Dict]] = defaultdict(list)
        today = datetime.now(timezone.utc).date()

        for game in games:
            if game.get("status") != "Final":
                continue
            game_date_str = (game.get("date") or "")[:10]
            try:
                game_date = datetime.fromisoformat(game_date_str).date()
            except Exception:
                continue
            if game_date > today:
                continue

            ht_id = game.get("home_team", {}).get("id")
            at_id = game.get("visitor_team", {}).get("id")

            game_entry = {
                "date":        game_date,
                "game_id":     game.get("id"),
                "home_team_id": ht_id,
                "away_team_id": at_id,
                "home_score":  game.get("home_team_score", 0),
                "away_score":  game.get("visitor_team_score", 0),
            }
            if ht_id:
                team_games[ht_id].append({**game_entry, "role": "home"})
            if at_id:
                team_games[at_id].append({**game_entry, "role": "away"})

        # Sort each team's games ascending by date
        for tid in team_games:
            team_games[tid].sort(key=lambda g: g["date"])

        features: Dict[int, Dict[str, Any]] = {}

        for tid, tgames in team_games.items():
            if not tgames:
                continue

            last_game = tgames[-1]
            last_date = last_game["date"]
            days_rest = (today - last_date).days
            back_to_back = days_rest == 1

            # Home/away record in last 10 games
            recent = tgames[-self.ROLLING_WINDOW:]
            home_wins = away_wins = home_played = away_played = 0
            pts_scored_list: List[float] = []
            pts_allowed_list: List[float] = []
            possessions_list: List[float] = []

            for g in recent:
                is_home = g["role"] == "home"
                won = (
                    g["home_score"] > g["away_score"]
                    if is_home
                    else g["away_score"] > g["home_score"]
                )
                pts_scored  = g["home_score"] if is_home else g["away_score"]
                pts_allowed = g["away_score"] if is_home else g["home_score"]

                # Rough possession estimate: (pts_scored + pts_allowed) / 2 * pace_factor
                # A very simple proxy using raw scores
                possessions = (pts_scored + pts_allowed) / 2.0

                pts_scored_list.append(pts_scored)
                pts_allowed_list.append(pts_allowed)
                possessions_list.append(possessions)

                if is_home:
                    home_played += 1
                    home_wins   += int(won)
                else:
                    away_played += 1
                    away_wins   += int(won)

            n = len(recent)
            avg_scored  = sum(pts_scored_list) / n if n else 0.0
            avg_allowed = sum(pts_allowed_list) / n if n else 0.0
            avg_poss    = sum(possessions_list) / n if n else 100.0

            # Ratings per 100 possessions
            off_rating = (avg_scored / avg_poss * 100) if avg_poss else 0.0
            def_rating = (avg_allowed / avg_poss * 100) if avg_poss else 0.0
            # Pace: possessions per 48 minutes – proxy via raw avg_poss scaled
            pace = avg_poss * 2.0  # rough scaling factor

            features[tid] = {
                "team_id":                 tid,
                "days_rest_since_last_game": days_rest,
                "back_to_back":             back_to_back,
                "home_record_last_10":      f"{home_wins}-{home_played - home_wins}",
                "away_record_last_10":      f"{away_wins}-{away_played - away_wins}",
                "offensive_rating":         round(off_rating, 2),
                "defensive_rating":         round(def_rating, 2),
                "pace":                     round(pace, 2),
                "last_game_date":           str(last_date),
            }

        return features

    # ------------------------------------------------------------------
    # BaseSource interface
    # ------------------------------------------------------------------

    def fetch(self) -> Dict:
        """Fetch teams, season games, injuries, and compute team features."""
        result: Dict[str, Any] = {
            "teams":    [],
            "games":    [],
            "injuries": [],
            "features": {},
        }

        # Teams
        try:
            result["teams"] = self.fetch_teams()
            self.logger.info(f"NBA: fetched {len(result['teams'])} teams")
        except Exception as exc:
            self.logger.warning(f"NBA teams fetch failed: {exc}")

        # Games for current season
        try:
            result["games"] = self.fetch_games(season=self.season)
            self.logger.info(f"NBA: fetched {len(result['games'])} games")
        except Exception as exc:
            self.logger.warning(f"NBA games fetch failed: {exc}")

        # Injury report (requires API key)
        try:
            result["injuries"] = self.fetch_player_injuries()
            self.logger.info(f"NBA: fetched {len(result['injuries'])} injury entries")
        except Exception as exc:
            self.logger.warning(f"NBA injury fetch failed: {exc}")

        # Compute features from game data
        if result["games"]:
            result["features"] = self._compute_team_features(result["games"])
            self.logger.info(
                f"NBA: computed features for {len(result['features'])} teams"
            )

        return result

    def validate(self, data: Dict) -> bool:
        if not isinstance(data, dict):
            return False
        if "games" not in data:
            self.logger.warning("NBA validation failed: 'games' key missing")
            return False
        return True

    def normalize(self, data: Dict) -> pd.DataFrame:
        """Return team-features DataFrame with one row per team."""
        features = data.get("features", {})
        if not features:
            return pd.DataFrame(columns=[
                "team_id", "days_rest_since_last_game", "back_to_back",
                "home_record_last_10", "away_record_last_10",
                "offensive_rating", "defensive_rating", "pace",
                "last_game_date",
            ])
        return pd.DataFrame(list(features.values()))

    def get_games_df(self, data: Dict) -> pd.DataFrame:
        """Return a normalised DataFrame of raw game records."""
        games = data.get("games", [])
        if not games:
            return pd.DataFrame(columns=[
                "game_id", "date", "season", "status",
                "home_team_id", "home_team_name", "home_score",
                "away_team_id", "away_team_name", "away_score",
            ])
        rows = []
        for g in games:
            rows.append({
                "game_id":        g.get("id"),
                "date":           (g.get("date") or "")[:10],
                "season":         g.get("season"),
                "status":         g.get("status"),
                "home_team_id":   g.get("home_team", {}).get("id"),
                "home_team_name": g.get("home_team", {}).get("full_name"),
                "home_score":     g.get("home_team_score"),
                "away_team_id":   g.get("visitor_team", {}).get("id"),
                "away_team_name": g.get("visitor_team", {}).get("full_name"),
                "away_score":     g.get("visitor_team_score"),
            })
        return pd.DataFrame(rows)

    def get_injuries_df(self, data: Dict) -> pd.DataFrame:
        """Return a normalised DataFrame of player injuries."""
        injuries = data.get("injuries", [])
        if not injuries:
            return pd.DataFrame(columns=[
                "player_id", "player_name", "team_id", "team_name",
                "status", "comment",
            ])
        rows = []
        for inj in injuries:
            player = inj.get("player", {})
            team   = inj.get("team", {})
            rows.append({
                "player_id":   player.get("id"),
                "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                "team_id":     team.get("id"),
                "team_name":   team.get("full_name"),
                "status":      inj.get("status"),
                "comment":     inj.get("notes") or inj.get("comment"),
            })
        return pd.DataFrame(rows)
