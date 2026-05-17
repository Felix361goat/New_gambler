import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def calculate_hockey_features(
    home_team: str,
    away_team: str,
    game_date,
    matches_df: Optional[pd.DataFrame],
    config: dict,
) -> dict:
    """
    Hockey-specific features: back-to-back, travel, goalie fatigue.
    """
    if isinstance(game_date, str):
        try:
            game_date = pd.Timestamp(game_date)
        except Exception:
            game_date = pd.Timestamp.now()

    features = {}
    features.update(_back_to_back_features(home_team, game_date, matches_df, "home"))
    features.update(_back_to_back_features(away_team, game_date, matches_df, "away"))
    features.update(_travel_features(home_team, away_team, matches_df))
    features.update(_home_away_split(home_team, away_team, matches_df))
    features["sport"] = "hockey"
    return features


def _back_to_back_features(team: str, game_date, df: Optional[pd.DataFrame], prefix: str) -> dict:
    """Back-to-back flag: played yesterday? How many rest days?"""
    if df is None or df.empty:
        return {
            f"{prefix}_back_to_back": False,
            f"{prefix}_days_rest": 7,
            f"{prefix}_b2b_penalty": 0.0,
            f"{prefix}_games_last_7d": 0,
        }

    team_games = df[
        (df.get("home_team", pd.Series()) == team) | (df.get("away_team", pd.Series()) == team)
    ].copy()

    if "date" in team_games.columns:
        team_games["date"] = pd.to_datetime(team_games["date"], errors="coerce")
        past_games = team_games[team_games["date"] < game_date].sort_values("date", ascending=False)
    else:
        past_games = pd.DataFrame()

    if past_games.empty:
        return {
            f"{prefix}_back_to_back": False,
            f"{prefix}_days_rest": 7,
            f"{prefix}_b2b_penalty": 0.0,
            f"{prefix}_games_last_7d": 0,
        }

    last_game_date = past_games.iloc[0]["date"]
    days_rest = (game_date - last_game_date).days if hasattr(game_date - last_game_date, "days") else 7
    back_to_back = days_rest <= 1

    # B2B penalty: research shows ~5-8% performance drop on 2nd game
    b2b_penalty = 0.07 if back_to_back else 0.0

    # Games in last 7 days
    week_cutoff = game_date - pd.Timedelta(days=7)
    games_last_7d = len(team_games[team_games.get("date", pd.Series()) >= week_cutoff]) if "date" in team_games.columns else 0

    return {
        f"{prefix}_back_to_back": back_to_back,
        f"{prefix}_days_rest": min(days_rest, 14),
        f"{prefix}_b2b_penalty": b2b_penalty,
        f"{prefix}_games_last_7d": games_last_7d,
    }


def _travel_features(home_team: str, away_team: str, df: Optional[pd.DataFrame]) -> dict:
    """Approximate travel distance for away team's previous game location."""
    # Without a city-coordinates DB for AHL/ECHL arenas, use previous game home/away as proxy
    if df is None or df.empty:
        return {"away_consecutive_road_games": 0, "away_road_trip_length": 0}

    away_games = df[df.get("away_team", pd.Series()) == away_team].copy()
    if "date" in away_games.columns:
        away_games = away_games.sort_values("date", ascending=False)

    # Count consecutive road games
    consecutive = 0
    for _, row in away_games.head(10).iterrows():
        if row.get("away_team") == away_team:
            consecutive += 1
        else:
            break

    return {
        "away_consecutive_road_games": consecutive,
        "away_road_trip_length": consecutive,
    }


def _home_away_split(home_team: str, away_team: str, df: Optional[pd.DataFrame]) -> dict:
    """Home and away performance split."""
    if df is None or df.empty:
        return {
            "home_team_home_win_rate": 0.5,
            "away_team_away_win_rate": 0.5,
        }

    def win_rate(team, location, last_n=20):
        if location == "home":
            games = df[df.get("home_team", pd.Series()) == team].head(last_n)
        else:
            games = df[df.get("away_team", pd.Series()) == team].head(last_n)

        if games.empty:
            return 0.5

        wins = 0
        for _, row in games.iterrows():
            hg = row.get("home_goals", row.get("home_score", 0)) or 0
            ag = row.get("away_goals", row.get("away_score", 0)) or 0
            if location == "home" and hg > ag:
                wins += 1
            elif location == "away" and ag > hg:
                wins += 1
        return round(wins / len(games), 4)

    return {
        "home_team_home_win_rate": win_rate(home_team, "home"),
        "away_team_away_win_rate": win_rate(away_team, "away"),
    }


def _empty_hockey() -> dict:
    return {
        "home_back_to_back": False,
        "home_days_rest": 7,
        "home_b2b_penalty": 0.0,
        "home_games_last_7d": 0,
        "away_back_to_back": False,
        "away_days_rest": 7,
        "away_b2b_penalty": 0.0,
        "away_games_last_7d": 0,
        "away_consecutive_road_games": 0,
        "away_road_trip_length": 0,
        "home_team_home_win_rate": 0.5,
        "away_team_away_win_rate": 0.5,
        "sport": "hockey",
    }
