import numpy as np
from typing import Optional


def calculate_squad_features(team_id: str, injury_data: list, market_values: dict) -> dict:
    if injury_data is None:
        injury_data = []
    if market_values is None:
        market_values = {}

    team_injuries = [p for p in injury_data if p.get("team") == team_id]
    injured_count = len(team_injuries)

    team_values = market_values.get(team_id, {})
    all_values = sorted(team_values.values(), reverse=True) if isinstance(team_values, dict) else []

    # Key player = top 20% by market value
    if all_values:
        threshold_idx = max(1, int(len(all_values) * 0.20))
        top_values = set(list(team_values.keys())[:threshold_idx]) if isinstance(team_values, dict) else set()
        injured_names = {p.get("player_name", "") for p in team_injuries}
        key_players_missing = bool(top_values & injured_names)

        total_value = sum(all_values)
        injured_value = sum(
            team_values.get(p.get("player_name", ""), 0)
            for p in team_injuries
            if isinstance(team_values, dict)
        )
        available_value = total_value - injured_value
        injury_impact = injured_value / total_value if total_value > 0 else 0.0

        # Depth score: how evenly distributed is squad value
        top11 = all_values[:11]
        depth_score = (np.mean(top11) / np.max(top11)) if top11 else 0.5
    else:
        key_players_missing = False
        available_value = 0.0
        injury_impact = min(injured_count * 0.1, 1.0)
        depth_score = 0.5

    return {
        "injured_players_count": injured_count,
        "key_players_missing": key_players_missing,
        "squad_depth_score": round(float(depth_score), 3),
        "available_squad_value": round(float(available_value), 2),
        "injury_impact_score": round(float(injury_impact), 3),
    }
