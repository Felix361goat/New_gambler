import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def calculate_basketball_lower_features(
    home_team: str,
    away_team: str,
    matches_df: Optional[pd.DataFrame],
    injury_data: Optional[list],
    roster: Optional[dict],
    config: dict,
) -> dict:
    """
    Lower-league basketball features.
    Small rosters = single player impact is outsized.
    """
    features = {}
    features.update(_player_impact(home_team, injury_data, roster, "home"))
    features.update(_player_impact(away_team, injury_data, roster, "away"))
    features.update(_rest_days(home_team, away_team, matches_df))
    features.update(_home_away_split(home_team, away_team, matches_df))
    features["sport"] = "basketball"
    return features


def _player_impact(team: str, injury_data: Optional[list], roster: Optional[dict], prefix: str) -> dict:
    """
    Single player impact — critical in small-roster leagues.
    If top scorer / best player is out, impact is huge.
    """
    if not injury_data:
        return {
            f"{prefix}_injured_count": 0,
            f"{prefix}_key_player_missing": False,
            f"{prefix}_player_impact_score": 1.0,
        }

    team_injuries = [p for p in injury_data if p.get("team") == team]
    injured_count = len(team_injuries)

    # Key player = anyone in top 30% of roster by usage/minutes
    team_roster = (roster or {}).get(team, [])
    roster_size = max(len(team_roster), 8)  # assume 8 minimum

    key_threshold = max(1, int(roster_size * 0.30))
    key_players = {p.get("name", "") for p in team_roster[:key_threshold]}
    injured_names = {p.get("player_name", p.get("name", "")) for p in team_injuries}
    key_missing = bool(key_players & injured_names)

    # Impact score: 1.0 = full strength, 0.0 = massively depleted
    # Small roster: each player = 1/roster_size of total strength
    impact = max(0.0, 1.0 - (injured_count / roster_size))
    if key_missing:
        impact *= 0.80  # additional 20% penalty for key player

    return {
        f"{prefix}_injured_count": injured_count,
        f"{prefix}_key_player_missing": key_missing,
        f"{prefix}_player_impact_score": round(impact, 3),
    }


def _rest_days(home_team: str, away_team: str, df: Optional[pd.DataFrame]) -> dict:
    if df is None or df.empty:
        return {"home_rest_days": 3, "away_rest_days": 3}

    now = pd.Timestamp.now()

    def get_rest(team):
        games = df[
            (df.get("home_team", pd.Series()) == team) | (df.get("away_team", pd.Series()) == team)
        ]
        if "date" in games.columns:
            past = games[pd.to_datetime(games["date"], errors="coerce") < now]
            if not past.empty:
                last = pd.to_datetime(past["date"], errors="coerce").max()
                return max(0, (now - last).days)
        return 3

    return {
        "home_rest_days": get_rest(home_team),
        "away_rest_days": get_rest(away_team),
    }


def _home_away_split(home_team: str, away_team: str, df: Optional[pd.DataFrame]) -> dict:
    if df is None or df.empty:
        return {"home_win_rate_home": 0.5, "away_win_rate_away": 0.5}

    def win_rate(team, loc, n=15):
        if loc == "home":
            games = df[df.get("home_team", pd.Series()) == team].head(n)
        else:
            games = df[df.get("away_team", pd.Series()) == team].head(n)
        if games.empty:
            return 0.5
        wins = sum(
            1 for _, r in games.iterrows()
            if (loc == "home" and (r.get("home_score", 0) or 0) > (r.get("away_score", 0) or 0)) or
               (loc == "away" and (r.get("away_score", 0) or 0) > (r.get("home_score", 0) or 0))
        )
        return round(wins / len(games), 4)

    return {
        "home_win_rate_home": win_rate(home_team, "home"),
        "away_win_rate_away": win_rate(away_team, "away"),
    }
