import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def calculate_movement_features(match_id: str, db, hours_lookback: int = 48) -> dict:
    try:
        with db._get_conn() as conn:
            cutoff = datetime.now() - timedelta(hours=hours_lookback)
            rows = conn.execute(
                """SELECT * FROM odds_snapshots
                   WHERE match_id = ? AND captured_at >= ?
                   ORDER BY captured_at ASC""",
                (match_id, cutoff.isoformat()),
            ).fetchall()

        if not rows or len(rows) < 2:
            return _empty_movement()

        rows = [dict(r) for r in rows]

        # Aggregate home odds across all bookmakers at each snapshot
        def avg_home_odds(snapshot):
            return snapshot.get("odds_home")

        opening = next((r for r in rows if avg_home_odds(r) is not None), None)
        current = next((r for r in reversed(rows) if avg_home_odds(r) is not None), None)

        if not opening or not current:
            return _empty_movement()

        op = avg_home_odds(opening)
        cu = avg_home_odds(current)

        if op is None or op == 0:
            return _empty_movement()

        movement_magnitude = (cu - op) / op  # fraction
        direction = "shortening" if cu < op else "drifting"

        # Bookmaker consensus: what % of unique bookmakers moved same direction
        bookmakers = {}
        for row in rows:
            bk = row.get("bookmaker")
            if bk and row.get("odds_home") is not None:
                if bk not in bookmakers:
                    bookmakers[bk] = {"first": row["odds_home"], "last": row["odds_home"]}
                else:
                    bookmakers[bk]["last"] = row["odds_home"]

        if len(bookmakers) > 1:
            shorter = sum(
                1 for v in bookmakers.values() if v["last"] < v["first"]
            )
            consensus = shorter / len(bookmakers) if direction == "shortening" else (len(bookmakers) - shorter) / len(bookmakers)
        else:
            consensus = 0.5

        # Sharp money: >70% consensus = sharp
        sharp_signal = 0.0
        if consensus >= 0.70:
            sharp_signal = 1.0 if direction == "shortening" else -1.0
        elif consensus >= 0.55:
            sharp_signal = 0.5 if direction == "shortening" else -0.5

        return {
            "opening_odds": round(op, 3),
            "current_odds": round(cu, 3),
            "movement_direction": direction,
            "movement_magnitude": round(movement_magnitude, 4),
            "bookmaker_consensus": round(consensus, 3),
            "sharp_money_signal": round(sharp_signal, 2),
        }
    except Exception as e:
        logger.error(f"calculate_movement_features failed for {match_id}: {e}")
        return _empty_movement()


def _empty_movement() -> dict:
    return {
        "opening_odds": 0.0,
        "current_odds": 0.0,
        "movement_direction": "unknown",
        "movement_magnitude": 0.0,
        "bookmaker_consensus": 0.5,
        "sharp_money_signal": 0.0,
    }
