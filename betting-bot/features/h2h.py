import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_h2h_features(home_team: str, away_team: str, matches_df: pd.DataFrame, limit: int = 10) -> dict:
    if matches_df is None or matches_df.empty:
        return _empty_h2h()

    mask = (
        ((matches_df["home_team"] == home_team) & (matches_df["away_team"] == away_team)) |
        ((matches_df["home_team"] == away_team) & (matches_df["away_team"] == home_team))
    )
    h2h = matches_df[mask].copy()

    if "date" in h2h.columns:
        h2h = h2h.sort_values("date", ascending=False)
    h2h = h2h.head(limit)

    if h2h.empty:
        return _empty_h2h()

    home_wins = away_wins = draws = 0
    total_goals = []

    for _, row in h2h.iterrows():
        hg = row.get("home_goals", 0) or 0
        ag = row.get("away_goals", 0) or 0
        total_goals.append(hg + ag)
        is_home = row.get("home_team") == home_team
        if hg > ag:
            if is_home: home_wins += 1
            else: away_wins += 1
        elif ag > hg:
            if is_home: away_wins += 1
            else: home_wins += 1
        else:
            draws += 1

    n = len(h2h)
    return {
        "h2h_home_win_rate": round(home_wins / n, 3),
        "h2h_away_win_rate": round(away_wins / n, 3),
        "h2h_draw_rate": round(draws / n, 3),
        "h2h_avg_total_goals": round(np.mean(total_goals), 3),
        "h2h_over25_rate": round(sum(1 for g in total_goals if g > 2.5) / n, 3),
        "h2h_sample_size": n,
    }


def _empty_h2h() -> dict:
    return {
        "h2h_home_win_rate": 0.33,
        "h2h_away_win_rate": 0.33,
        "h2h_draw_rate": 0.34,
        "h2h_avg_total_goals": 2.5,
        "h2h_over25_rate": 0.50,
        "h2h_sample_size": 0,
    }
