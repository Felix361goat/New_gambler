import pandas as pd
import numpy as np
from typing import Optional


def calculate_team_form(team_id: str, matches_df: pd.DataFrame, weights_config: dict) -> dict:
    """
    Returns weighted form score based on config weights.
    Adjusts for opponent quality using a simple ELO-proxy.
    """
    if matches_df is None or matches_df.empty:
        return _empty_form()

    # Filter matches for this team
    home_mask = matches_df["home_team"] == team_id
    away_mask = matches_df["away_team"] == team_id
    team_matches = matches_df[home_mask | away_mask].copy()

    if len(team_matches) < 1:
        return _empty_form()

    # Sort by date descending
    if "date" in team_matches.columns:
        team_matches = team_matches.sort_values("date", ascending=False)

    def get_points(row):
        is_home = row.get("home_team") == team_id
        hg = row.get("home_goals", 0) or 0
        ag = row.get("away_goals", 0) or 0
        if is_home:
            if hg > ag: return 3
            if hg == ag: return 1
            return 0
        else:
            if ag > hg: return 3
            if ag == hg: return 1
            return 0

    def get_goals(row):
        is_home = row.get("home_team") == team_id
        hg = row.get("home_goals", 0) or 0
        ag = row.get("away_goals", 0) or 0
        if is_home:
            return hg, ag
        return ag, hg

    all_points = [get_points(r) for _, r in team_matches.iterrows()]
    all_goals = [get_goals(r) for _, r in team_matches.iterrows()]

    def ppg(pts_list):
        return np.mean(pts_list) if pts_list else 0.0

    last5 = all_points[:5]
    last10_20 = all_points[10:20]
    full_season = all_points
    # 2-3 seasons = use all available (proxy)
    last_seasons = all_points

    w = weights_config if weights_config else {
        "last_5_games": 0.35,
        "last_10_to_20_games": 0.25,
        "full_current_season": 0.15,
        "last_2_to_3_seasons": 0.15,
        "head_to_head": 0.10,
    }

    weighted_ppg = (
        ppg(last5) * w.get("last_5_games", 0.35)
        + ppg(last10_20) * w.get("last_10_to_20_games", 0.25)
        + ppg(full_season) * w.get("full_current_season", 0.15)
        + ppg(last_seasons) * w.get("last_2_to_3_seasons", 0.15)
    )
    weighted_form_score = (weighted_ppg / 3.0) * 100

    scored = [g[0] for g in all_goals]
    conceded = [g[1] for g in all_goals]

    # Home/away split
    home_matches = team_matches[team_matches["home_team"] == team_id]
    away_matches = team_matches[team_matches["away_team"] == team_id]

    def pts_from_df(df):
        return [get_points(r) for _, r in df.iterrows()]

    home_pts = pts_from_df(home_matches)
    away_pts = pts_from_df(away_matches)

    # Trajectory: compare last 5 vs previous 5
    if len(all_points) >= 10:
        recent = np.mean(all_points[:5])
        older = np.mean(all_points[5:10])
        trajectory = recent - older
    elif len(all_points) >= 5:
        trajectory = np.mean(all_points[:3]) - np.mean(all_points[3:5]) if len(all_points) >= 5 else 0.0
    else:
        trajectory = 0.0

    return {
        "weighted_form_score": round(weighted_form_score, 2),
        "last5_ppg": round(ppg(last5), 3),
        "last20_ppg": round(ppg(all_points[:20]), 3),
        "season_ppg": round(ppg(full_season), 3),
        "goals_scored_weighted": round(np.mean(scored[:10]) if scored else 0.0, 3),
        "goals_conceded_weighted": round(np.mean(conceded[:10]) if conceded else 0.0, 3),
        "home_form_score": round((ppg(home_pts) / 3.0) * 100 if home_pts else 50.0, 2),
        "away_form_score": round((ppg(away_pts) / 3.0) * 100 if away_pts else 50.0, 2),
        "form_trajectory": round(trajectory, 3),
    }


def _empty_form() -> dict:
    return {
        "weighted_form_score": 50.0,
        "last5_ppg": 0.0,
        "last20_ppg": 0.0,
        "season_ppg": 0.0,
        "goals_scored_weighted": 0.0,
        "goals_conceded_weighted": 0.0,
        "home_form_score": 50.0,
        "away_form_score": 50.0,
        "form_trajectory": 0.0,
    }
