import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SURFACE_COL = "surface"
SURFACES = ["clay", "hard", "grass", "carpet"]


def calculate_tennis_features(
    player: str,
    opponent: str,
    surface: str,
    matches_df: Optional[pd.DataFrame],
    config: dict,
) -> dict:
    """
    Surface-aware tennis feature calculation.
    Covers: surface win rate, form, H2H by surface, tournament fatigue.
    """
    surface = surface.lower()
    if matches_df is None or matches_df.empty:
        return _empty_tennis()

    features = {}

    # Surface-specific win rate (last 24 months)
    features.update(_surface_win_rate(player, surface, matches_df))

    # Recent form (last 5 and 10 matches, all surfaces + surface-specific)
    features.update(_recent_form(player, surface, matches_df))

    # H2H on this surface
    features.update(_h2h_surface(player, opponent, surface, matches_df))

    # Tournament fatigue
    features.update(_tournament_fatigue(player, matches_df))

    # Serve stats (if available)
    features.update(_serve_stats(player, surface, matches_df))

    # Ranking band (50-200 = target range)
    features.update(_ranking_features(player, matches_df))

    features["sport"] = "tennis"
    features["surface"] = surface

    return features


def _surface_win_rate(player: str, surface: str, df: pd.DataFrame) -> dict:
    """Win rate on specific surface over last 24 months."""
    from datetime import datetime, timedelta
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=730)

    # Filter player matches on surface in last 24 months
    player_matches = df[
        ((df.get("player1", pd.Series()) == player) | (df.get("player2", pd.Series()) == player)) &
        (df.get(SURFACE_COL, pd.Series()) == surface)
    ]
    if "date" in player_matches.columns:
        player_matches = player_matches[player_matches["date"] >= cutoff]

    if len(player_matches) == 0:
        return {
            f"surface_{surface}_win_rate_24m": 0.5,
            f"surface_{surface}_matches_24m": 0,
        }

    wins = 0
    total = len(player_matches)
    for _, row in player_matches.iterrows():
        winner = row.get("winner", row.get("player1", ""))
        if winner == player:
            wins += 1

    return {
        f"surface_{surface}_win_rate_24m": round(wins / total, 4),
        f"surface_{surface}_matches_24m": total,
    }


def _recent_form(player: str, surface: str, df: pd.DataFrame) -> dict:
    """Last 5 and 10 matches form, weighted by recency."""
    player_matches = df[
        (df.get("player1", pd.Series()) == player) | (df.get("player2", pd.Series()) == player)
    ].copy()

    if "date" in player_matches.columns:
        player_matches = player_matches.sort_values("date", ascending=False)

    def win_rate(rows, n):
        subset = rows.head(n)
        if len(subset) == 0:
            return 0.5
        wins = sum(1 for _, r in subset.iterrows() if r.get("winner", r.get("player1", "")) == player)
        return round(wins / len(subset), 4)

    # Surface-specific last 10
    surface_matches = player_matches[player_matches.get(SURFACE_COL, pd.Series()) == surface]

    return {
        "form_last5_all": win_rate(player_matches, 5),
        "form_last10_all": win_rate(player_matches, 10),
        f"form_last10_{surface}": win_rate(surface_matches, 10),
        "form_trajectory": _form_trajectory(player, player_matches),
    }


def _form_trajectory(player: str, player_matches: pd.DataFrame) -> float:
    """Positive = improving, negative = declining. Compare last 3 vs previous 3."""
    if len(player_matches) < 6:
        return 0.0
    recent = player_matches.head(3)
    older = player_matches.iloc[3:6]

    def win_rate(rows):
        if len(rows) == 0:
            return 0.5
        wins = sum(1 for _, r in rows.iterrows() if r.get("winner", r.get("player1", "")) == player)
        return wins / len(rows)

    return round(win_rate(recent) - win_rate(older), 4)


def _h2h_surface(player: str, opponent: str, surface: str, df: pd.DataFrame) -> dict:
    """Head-to-head record on this surface."""
    h2h = df[
        (
            ((df.get("player1", pd.Series()) == player) & (df.get("player2", pd.Series()) == opponent)) |
            ((df.get("player1", pd.Series()) == opponent) & (df.get("player2", pd.Series()) == player))
        ) &
        (df.get(SURFACE_COL, pd.Series()) == surface)
    ]

    total = len(h2h)
    if total == 0:
        return {"h2h_surface_win_rate": 0.5, "h2h_surface_matches": 0}

    wins = sum(1 for _, r in h2h.iterrows() if r.get("winner", r.get("player1", "")) == player)
    return {
        "h2h_surface_win_rate": round(wins / total, 4),
        "h2h_surface_matches": total,
    }


def _tournament_fatigue(player: str, df: pd.DataFrame) -> dict:
    """Matches played in last 14 and 30 days."""
    player_matches = df[
        (df.get("player1", pd.Series()) == player) | (df.get("player2", pd.Series()) == player)
    ].copy()

    if "date" not in player_matches.columns:
        return {"matches_last_14d": 0, "matches_last_30d": 0, "fatigue_flag": False}

    now = pd.Timestamp.now()
    last14 = player_matches[player_matches["date"] >= now - pd.Timedelta(days=14)]
    last30 = player_matches[player_matches["date"] >= now - pd.Timedelta(days=30)]

    n14 = len(last14)
    n30 = len(last30)
    # Flag if >6 matches in 14 days or >12 in 30 days
    fatigue = n14 > 6 or n30 > 12

    return {
        "matches_last_14d": n14,
        "matches_last_30d": n30,
        "fatigue_flag": fatigue,
    }


def _serve_stats(player: str, surface: str, df: pd.DataFrame) -> dict:
    """First serve % and break point conversion by surface (if available)."""
    # These columns may not always be present
    serve_cols = ["first_serve_pct", "bp_conversion_pct", "ace_rate"]
    player_matches = df[
        (df.get("player1", pd.Series()) == player) | (df.get("player2", pd.Series()) == player)
    ]
    if SURFACE_COL in player_matches.columns:
        player_matches = player_matches[player_matches[SURFACE_COL] == surface]

    result = {}
    for col in serve_cols:
        if col in player_matches.columns:
            val = player_matches[col].mean()
            result[f"{col}_{surface}"] = round(float(val), 4) if not pd.isna(val) else 0.0
        else:
            result[f"{col}_{surface}"] = 0.0

    return result


def _ranking_features(player: str, df: pd.DataFrame) -> dict:
    """Current ranking and whether in target range (50-200)."""
    # Look for ranking column
    if "ranking" in df.columns:
        player_rows = df[df.get("player1", pd.Series()) == player]
        if not player_rows.empty:
            ranking = player_rows.iloc[0].get("ranking", 999)
            return {
                "ranking": int(ranking),
                "in_target_range": 50 <= ranking <= 200,
                "ranking_band": _ranking_band(ranking),
            }
    return {"ranking": 999, "in_target_range": False, "ranking_band": "unknown"}


def _ranking_band(ranking: int) -> str:
    if ranking <= 50:
        return "top50"
    if ranking <= 100:
        return "50-100"
    if ranking <= 200:
        return "100-200"
    return "200+"


def _empty_tennis() -> dict:
    return {
        "surface_clay_win_rate_24m": 0.5,
        "surface_hard_win_rate_24m": 0.5,
        "surface_grass_win_rate_24m": 0.5,
        "surface_clay_matches_24m": 0,
        "surface_hard_matches_24m": 0,
        "surface_grass_matches_24m": 0,
        "form_last5_all": 0.5,
        "form_last10_all": 0.5,
        "form_last10_hard": 0.5,
        "form_trajectory": 0.0,
        "h2h_surface_win_rate": 0.5,
        "h2h_surface_matches": 0,
        "matches_last_14d": 0,
        "matches_last_30d": 0,
        "fatigue_flag": False,
        "first_serve_pct_hard": 0.0,
        "bp_conversion_pct_hard": 0.0,
        "ace_rate_hard": 0.0,
        "ranking": 999,
        "in_target_range": False,
        "ranking_band": "unknown",
        "sport": "tennis",
        "surface": "hard",
    }
