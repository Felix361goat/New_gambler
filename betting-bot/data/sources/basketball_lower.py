"""Basketball lower-tier data source.

Covers: Baltic Basketball League, Romanian Basketball League, and
other lower European basketball leagues.

Data sourced from:
- SofaScore scraping (primary): live scores, lineups, player stats
  https://www.sofascore.com
- Flashscore scraping (backup): results and schedules
  https://www.flashscore.com

WHY Poisson is disabled for basketball:
  Basketball point totals (typically 60-100+ per team) are far outside the
  low-count Poisson regime (mean 1-3) where Dixon-Coles is calibrated.  The
  negative binomial or normal distribution fits better, but XGBoost and ELO
  handle the regression task adequately without a separate count model.

Key features:
- Player impact score: roster-weighted quality accounting for injuries
- Key player missing flag: if top-salary / top-minute player is absent
- Roster size: smaller rosters make individual absences more impactful
"""

import logging
from typing import Optional

import pandas as pd
import requests

from data.sources.base_source import BaseSource

logger = logging.getLogger(__name__)

# Threshold: if a player accounts for >=25% of team's estimated impact,
# their absence is flagged as "key player missing".
KEY_PLAYER_IMPACT_THRESHOLD = 0.25


class BasketballLowerSource(BaseSource):
    """Fetch and normalize lower-tier European basketball data."""

    def __init__(self, config: Optional[dict] = None):
        self.config  = config or {}
        self.timeout = (config or {}).get("data_sources", {}).get("timeout_seconds", 10)
        self._sofascore_base = (
            (config or {})
            .get("data_sources", {})
            .get("sofascore_base_url", "https://www.sofascore.com")
        )
        self._flashscore_base = (
            (config or {})
            .get("data_sources", {})
            .get("flashscore_base_url", "https://www.flashscore.com")
        )

    # ------------------------------------------------------------------
    # BaseSource interface
    # ------------------------------------------------------------------

    def fetch(self) -> dict:
        """Fetch upcoming basketball games from SofaScore."""
        # Baltic Basketball League SofaScore tournament ID: 9028 (example)
        # Romanian Basketball League: 6225
        # These IDs need to be maintained / updated per season.
        tournament_ids = self.config.get("basketball_sofascore_ids", [9028, 6225])
        results = {}
        for tid in tournament_ids:
            try:
                url = f"{self._sofascore_base}/api/v1/unique-tournament/{tid}/events/next/0"
                resp = requests.get(
                    url,
                    timeout=self.timeout,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                resp.raise_for_status()
                results[tid] = resp.json()
            except Exception as e:
                logger.warning(f"BasketballLowerSource.fetch() failed for tournament {tid}: {e}")
        return results

    def validate(self, data: dict) -> bool:
        return isinstance(data, dict) and len(data) > 0

    def normalize(self, data: dict) -> pd.DataFrame:
        """Convert raw SofaScore events to standard match format."""
        rows = []
        for tid, payload in data.items():
            for event in payload.get("events", []):
                try:
                    home = event.get("homeTeam", {}).get("name", "")
                    away = event.get("awayTeam", {}).get("name", "")
                    rows.append({
                        "match_id":   str(event.get("id", "")),
                        "date":       _epoch_to_date(event.get("startTimestamp")),
                        "league":     f"basketball_sofascore_{tid}",
                        "home_team":  home,
                        "away_team":  away,
                        "home_goals": event.get("homeScore", {}).get("current"),
                        "away_goals": event.get("awayScore", {}).get("current"),
                        "status":     event.get("status", {}).get("description", ""),
                        "sport":      "basketball",
                    })
                except Exception as e:
                    logger.debug(f"Basketball event parse error: {e}")
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Player impact calculation
    # ------------------------------------------------------------------

    def calculate_player_impact(
        self,
        team: str,
        injury_data: list,
        roster: dict,
    ) -> dict:
        """Estimate team strength adjustment based on roster availability.

        Args:
            team:         Team name.
            injury_data:  List of dicts with keys: team, player_name,
                          position, impact_weight (0-1).
            roster:       Dict mapping player_name → relative_impact (0-1).
                          Higher = more important to team performance.
                          If empty, all players assumed equal weight.

        Returns:
            Dict with:
            - player_impact_score (float 0-1): 1.0 = full squad, lower = injuries hurt
            - key_player_missing (bool): True if top-impact player is absent
            - injured_count (int): Number of injured/suspended players for this team
            - available_roster_pct (float): Share of roster available
        """
        if not injury_data and not roster:
            return {
                "player_impact_score": 1.0,
                "key_player_missing":  False,
                "injured_count":       0,
                "available_roster_pct": 1.0,
            }

        # Build injury set for this team
        injured_players: set = set()
        for entry in (injury_data or []):
            if entry.get("team", "") == team:
                injured_players.add(entry.get("player_name", ""))

        injured_count = len(injured_players)

        if not roster:
            # No roster data: estimate linearly by injured count vs assumed 12-man squad
            squad_size = max(injured_count + 5, 12)
            available_pct = max(0.0, (squad_size - injured_count) / squad_size)
            return {
                "player_impact_score": round(available_pct, 4),
                "key_player_missing":  False,  # Cannot determine without roster data
                "injured_count":       injured_count,
                "available_roster_pct": round(available_pct, 4),
            }

        # Calculate impact-weighted availability
        total_impact = sum(roster.values()) or 1.0
        lost_impact  = sum(
            roster.get(p, 0.0) for p in injured_players if p in roster
        )
        remaining_impact = max(0.0, total_impact - lost_impact)
        impact_score = round(remaining_impact / total_impact, 4)

        # Key player missing: any single injured player whose weight >= threshold
        key_missing = any(
            roster.get(p, 0.0) / total_impact >= KEY_PLAYER_IMPACT_THRESHOLD
            for p in injured_players
        )

        squad_size    = len(roster)
        available_pct = round((squad_size - injured_count) / max(squad_size, 1), 4)

        return {
            "player_impact_score":  impact_score,
            "key_player_missing":   key_missing,
            "injured_count":        injured_count,
            "available_roster_pct": available_pct,
        }

    # ------------------------------------------------------------------
    # Schedule / results helpers
    # ------------------------------------------------------------------

    def get_recent_results(
        self,
        team: str,
        historical_matches: Optional[pd.DataFrame],
        n: int = 10,
    ) -> pd.DataFrame:
        """Return the n most recent completed matches for a team."""
        if historical_matches is None or historical_matches.empty:
            return pd.DataFrame()

        df = historical_matches
        df.columns = [c.lower() for c in df.columns]

        mask = (
            (df.get("home_team", pd.Series(dtype=str)) == team) |
            (df.get("away_team", pd.Series(dtype=str)) == team)
        )
        team_matches = df[mask]

        if "date" in team_matches.columns:
            team_matches = team_matches.sort_values("date", ascending=False)

        return team_matches.head(n)

    def get_days_rest(
        self,
        team: str,
        game_date: str,
        historical_matches: Optional[pd.DataFrame],
    ) -> int:
        """Return calendar days since the team's last completed game.

        Returns 14 if no prior game is found (assume fully rested).
        """
        from datetime import datetime
        if historical_matches is None or historical_matches.empty:
            return 14

        try:
            target = datetime.strptime(game_date[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return 14

        df = historical_matches.copy()
        df.columns = [c.lower() for c in df.columns]

        mask = (
            (df.get("home_team", pd.Series(dtype=str)) == team) |
            (df.get("away_team", pd.Series(dtype=str)) == team)
        )
        team_df = df[mask]

        if team_df.empty or "date" not in team_df.columns:
            return 14

        team_df = team_df.copy()
        team_df["_date"] = pd.to_datetime(team_df["date"], errors="coerce").dt.date
        past = team_df[team_df["_date"] < target].dropna(subset=["_date"])

        if past.empty:
            return 14

        last_game = past["_date"].max()
        return int((target - last_game).days)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _epoch_to_date(epoch: Optional[int]) -> str:
    """Convert Unix epoch seconds to ISO date string."""
    if not epoch:
        return ""
    from datetime import datetime, timezone
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return ""
