import logging

logger = logging.getLogger(__name__)


def get_referee_modifier(referee_name: str, league: str, db) -> dict:
    if not referee_name:
        return _neutral()

    try:
        with db._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM referee_stats WHERE referee_name = ? AND league = ?",
                (referee_name, league),
            ).fetchone()

        if not row:
            logger.info(f"Unknown referee: {referee_name} in {league} — tracking for future use")
            _track_unknown_referee(referee_name, league, db)
            return _neutral()

        row = dict(row)
        sample = row.get("sample_size", 0) or 0
        confidence = min(1.0, sample / 50.0)

        return {
            "goals_modifier": round((row.get("avg_goals_per_game", 2.5) - 2.5) * confidence, 3),
            "cards_modifier": round((row.get("avg_cards_per_game", 4.0) - 4.0) * confidence, 3),
            "penalty_modifier": round((row.get("avg_penalties_per_game", 0.25) - 0.25) * confidence, 3),
            "home_bias": round((row.get("home_bias_score", 0.0) or 0.0) * confidence, 3),
            "confidence": round(confidence, 3),
        }
    except Exception as e:
        logger.error(f"get_referee_modifier failed: {e}")
        return _neutral()


def _neutral() -> dict:
    return {
        "goals_modifier": 0.0,
        "cards_modifier": 0.0,
        "penalty_modifier": 0.0,
        "home_bias": 0.0,
        "confidence": 0.0,
    }


def _track_unknown_referee(name: str, league: str, db):
    try:
        with db._get_conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO referee_stats
                   (referee_name, league, avg_goals_per_game, avg_cards_per_game,
                    avg_penalties_per_game, home_bias_score, sample_size)
                   VALUES (?, ?, 2.5, 4.0, 0.25, 0.0, 0)""",
                (name, league),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"_track_unknown_referee failed: {e}")


def update_referee_stats(referee_name: str, league: str, match_stats: dict, db):
    """Call after each settled match to update referee statistics."""
    try:
        with db._get_conn() as conn:
            existing = conn.execute(
                "SELECT * FROM referee_stats WHERE referee_name = ? AND league = ?",
                (referee_name, league),
            ).fetchone()

            goals = match_stats.get("total_goals", 2.5)
            cards = match_stats.get("total_cards", 4.0)
            penalties = match_stats.get("total_penalties", 0.0)
            home_advantage = match_stats.get("home_advantage", 0.0)

            if not existing:
                conn.execute(
                    """INSERT INTO referee_stats
                       (referee_name, league, avg_goals_per_game, avg_cards_per_game,
                        avg_penalties_per_game, home_bias_score, sample_size, last_updated)
                       VALUES (?, ?, ?, ?, ?, ?, 1, date('now'))""",
                    (referee_name, league, goals, cards, penalties, home_advantage),
                )
            else:
                row = dict(existing)
                n = row["sample_size"]
                new_n = n + 1
                def update_avg(old, new, count):
                    return (old * count + new) / (count + 1)
                conn.execute(
                    """UPDATE referee_stats SET
                       avg_goals_per_game = ?, avg_cards_per_game = ?,
                       avg_penalties_per_game = ?, home_bias_score = ?,
                       sample_size = ?, last_updated = date('now')
                       WHERE referee_name = ? AND league = ?""",
                    (
                        update_avg(row["avg_goals_per_game"] or 2.5, goals, n),
                        update_avg(row["avg_cards_per_game"] or 4.0, cards, n),
                        update_avg(row["avg_penalties_per_game"] or 0.25, penalties, n),
                        update_avg(row["home_bias_score"] or 0.0, home_advantage, n),
                        new_n, referee_name, league,
                    ),
                )
            conn.commit()
    except Exception as e:
        logger.error(f"update_referee_stats failed: {e}")
