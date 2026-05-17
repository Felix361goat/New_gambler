import pandas as pd
import numpy as np


def calculate_xg_features(team_id: str, matches_with_xg: pd.DataFrame) -> dict:
    """
    Teams consistently scoring > xG likely to regress.
    Teams scoring < xG are 'due' — real predictive signal.
    """
    if matches_with_xg is None or matches_with_xg.empty:
        return _empty_xg()

    home_mask = matches_with_xg["home_team"] == team_id
    away_mask = matches_with_xg["away_team"] == team_id
    team_df = matches_with_xg[home_mask | away_mask].copy()

    if team_df.empty:
        return _empty_xg()

    if "date" in team_df.columns:
        team_df = team_df.sort_values("date", ascending=False)

    xg_scored, xg_conceded, actual_scored = [], [], []
    npxg_scored_list = []

    for _, row in team_df.iterrows():
        is_home = row.get("home_team") == team_id
        if is_home:
            xg_scored.append(row.get("xg_home", 0) or 0)
            xg_conceded.append(row.get("xg_away", 0) or 0)
            npxg_scored_list.append(row.get("npxg_home", row.get("xg_home", 0)) or 0)
            actual_scored.append(row.get("home_goals", 0) or 0)
        else:
            xg_scored.append(row.get("xg_away", 0) or 0)
            xg_conceded.append(row.get("xg_home", 0) or 0)
            npxg_scored_list.append(row.get("npxg_away", row.get("xg_away", 0)) or 0)
            actual_scored.append(row.get("away_goals", 0) or 0)

    roll10 = lambda lst: np.mean(lst[:10]) if lst else 0.0

    xg_sc_r10 = roll10(xg_scored)
    xg_co_r10 = roll10(xg_conceded)
    xg_diff = xg_sc_r10 - xg_co_r10

    actual_r10 = np.mean(actual_scored[:10]) if actual_scored else 0.0
    xg_overperf = actual_r10 - xg_sc_r10

    # Trend: compare last 5 vs previous 5
    if len(xg_scored) >= 10:
        trend = np.mean(xg_scored[:5]) - np.mean(xg_scored[5:10])
    elif len(xg_scored) >= 5:
        trend = np.mean(xg_scored[:3]) - np.mean(xg_scored[3:5]) if len(xg_scored) >= 5 else 0.0
    else:
        trend = 0.0

    return {
        "xg_scored_rolling10": round(xg_sc_r10, 3),
        "xg_conceded_rolling10": round(xg_co_r10, 3),
        "xg_difference": round(xg_diff, 3),
        "xg_overperformance": round(xg_overperf, 3),
        "xg_trend": round(trend, 3),
        "npxg_scored": round(roll10(npxg_scored_list), 3),
    }


def _empty_xg() -> dict:
    return {
        "xg_scored_rolling10": 0.0,
        "xg_conceded_rolling10": 0.0,
        "xg_difference": 0.0,
        "xg_overperformance": 0.0,
        "xg_trend": 0.0,
        "npxg_scored": 0.0,
    }
