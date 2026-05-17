"""
Preference utilities — league priority and favorite club handling.
Used by filter.py for LAYER 3 and LAYER 5 logic.
"""
from typing import Optional


def get_league_priority(league: str, priority_list: list) -> int:
    """Returns priority index (lower = higher priority). Unknown leagues get lowest priority."""
    try:
        return priority_list.index(league)
    except ValueError:
        return len(priority_list)


def is_favorite_club_match(home_team: str, away_team: str, favorite_clubs: list) -> bool:
    return any(club in [home_team, away_team] for club in favorite_clubs)


def apply_ev_override(base_threshold: float, override_factor: float) -> float:
    """Returns adjusted EV threshold for favorite club matches."""
    return round(base_threshold * override_factor, 6)


def format_bet_for_display(bet: dict) -> str:
    """Human-readable one-liner for a bet."""
    market = bet.get("market", "")
    odds = bet.get("bookmaker_odds", 0)
    ev = bet.get("ev_score", 0) * 100
    return (
        f"{bet.get('home_team')} vs {bet.get('away_team')} | "
        f"{market} @ {odds:.2f} | EV: +{ev:.1f}%"
    )
