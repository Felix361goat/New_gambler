from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# City coordinates for travel distance approximation (lat, lon)
CITY_COORDS = {
    "Arsenal": (51.5549, -0.1084),
    "Chelsea": (51.4816, -0.1910),
    "Manchester City": (53.4831, -2.2004),
    "Manchester United": (53.4631, -2.2913),
    "Liverpool": (53.4308, -2.9608),
    "Tottenham": (51.6042, -0.0665),
    "default": (51.5074, -0.1278),
}


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    import math
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def calculate_context_features(match: dict, team_schedule: dict) -> dict:
    home_team = match.get("home_team", "")
    away_team = match.get("away_team", "")
    match_date = match.get("match_date")

    if isinstance(match_date, str):
        try:
            match_date = datetime.fromisoformat(match_date)
        except Exception:
            match_date = None

    def days_since_last(team_name):
        schedule = team_schedule.get(team_name, [])
        if not schedule or not match_date:
            return 7
        past = [d for d in schedule if d < match_date]
        if not past:
            return 7
        last = max(past)
        return (match_date - last).days

    def fixture_congestion(team_name, days=21):
        schedule = team_schedule.get(team_name, [])
        if not schedule or not match_date:
            return 0.0
        cutoff = match_date
        from datetime import timedelta
        start = match_date - timedelta(days=days)
        count = sum(1 for d in schedule if start <= d < cutoff)
        return count / days * 21

    home_coords = CITY_COORDS.get(home_team, CITY_COORDS["default"])
    away_coords = CITY_COORDS.get(away_team, CITY_COORDS["default"])
    travel_km = _haversine_km(*away_coords, *home_coords)

    # Match importance: use table position hints from match dict
    home_position = match.get("home_position", 10)
    away_position = match.get("away_position", 10)
    total_teams = match.get("total_teams", 20)

    # Relegation zone = bottom 3, title race = top 3
    def importance(pos):
        if pos <= 3 or pos >= total_teams - 2:
            return 1.0
        if pos <= 6 or pos >= total_teams - 5:
            return 0.7
        return 0.4

    importance_score = max(importance(home_position), importance(away_position))

    return {
        "days_since_last_game_home": days_since_last(home_team),
        "days_since_last_game_away": days_since_last(away_team),
        "fixture_congestion_home": round(fixture_congestion(home_team), 3),
        "fixture_congestion_away": round(fixture_congestion(away_team), 3),
        "travel_distance_km": round(travel_km, 1),
        "match_importance_score": round(importance_score, 2),
        "intl_players_returning_home": match.get("intl_players_returning_home", 0),
        "intl_players_returning_away": match.get("intl_players_returning_away", 0),
    }
