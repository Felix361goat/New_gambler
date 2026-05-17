import logging
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class PerformanceTracker:
    def __init__(self, db, config: dict):
        self.db = db
        self.config = config
        self.starting_bankroll = config.get("betting", {}).get("bankroll_paper", 1000.0)

    def get_current_bankroll(self) -> float:
        summary = self.db.get_performance_summary(days=3650)
        total_pnl = summary.get("total_pnl") or 0.0
        return self.starting_bankroll + total_pnl

    def get_roi(self, days: int = 30) -> float:
        summary = self.db.get_performance_summary(days=days)
        return summary.get("roi", 0.0)

    def get_settled_bet_count(self) -> int:
        try:
            with self.db._get_conn() as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM results"
                ).fetchone()[0]
                return count
        except Exception as e:
            logger.error(f"get_settled_bet_count failed: {e}")
            return 0

    def is_ready_for_live(self) -> bool:
        min_bets = self.config.get("betting", {}).get("min_paper_bets_before_live", 200)
        min_roi = 3.0
        settled = self.get_settled_bet_count()
        roi = self.get_roi(days=3650)
        return settled >= min_bets and roi > min_roi

    def calculate_max_drawdown(self, days: int = 90) -> float:
        try:
            with self.db._get_conn() as conn:
                rows = conn.execute(
                    """SELECT pnl_cumulative FROM performance
                       WHERE date >= date('now', ?)
                       ORDER BY date""",
                    (f"-{days} days",),
                ).fetchall()
            if not rows:
                return 0.0
            cumulative = [r[0] or 0 for r in rows]
            peak = cumulative[0]
            max_dd = 0.0
            for val in cumulative:
                peak = max(peak, val)
                dd = peak - val
                max_dd = max(max_dd, dd)
            return round(max_dd, 2)
        except Exception as e:
            logger.error(f"calculate_max_drawdown failed: {e}")
            return 0.0

    def check_clv_gate(self, db, min_bets: int = 30) -> tuple[bool, Optional[float]]:
        """
        Checks the rolling average CLV over the last `min_bets` settled bets.

        Returns:
            (True, avg_clv)  if avg_clv >= 0 — model is finding value.
            (False, avg_clv) if avg_clv < 0  — model is consistently overpaying.
            (True, None)     if fewer than `min_bets` settled bets exist yet
                             (not enough data to make a judgment).
        """
        try:
            with db._get_conn() as conn:
                rows = conn.execute(
                    """SELECT r.clv_score
                       FROM results r
                       WHERE r.clv_score IS NOT NULL
                       ORDER BY r.settled_at DESC
                       LIMIT ?""",
                    (min_bets,),
                ).fetchall()

            if len(rows) < min_bets:
                logger.info(
                    f"CLV gate: only {len(rows)} settled bets with CLV scores "
                    f"(need {min_bets}) — skipping gate check."
                )
                return True, None

            clv_values = [r[0] for r in rows]
            avg_clv = round(sum(clv_values) / len(clv_values), 6)
            passed = avg_clv >= 0.0
            return passed, avg_clv

        except Exception as e:
            logger.error(f"check_clv_gate failed: {e}")
            return True, None  # Fail open — don't block on DB errors

    def get_full_summary(self) -> dict:
        summary = self.db.get_performance_summary(days=3650)
        settled = self.get_settled_bet_count()
        won = summary.get("won") or 0
        win_rate = (won / settled * 100) if settled > 0 else 0.0
        return {
            "bankroll": round(self.get_current_bankroll(), 2),
            "total_pnl": round(summary.get("total_pnl") or 0.0, 2),
            "roi": round(summary.get("roi") or 0.0, 2),
            "settled_bets": settled,
            "won": won,
            "lost": summary.get("lost") or 0,
            "win_rate": round(win_rate, 1),
            "avg_ev": round(summary.get("avg_ev") or 0.0, 4),
            "avg_confidence": round(summary.get("avg_confidence") or 0.0, 1),
            "max_drawdown": self.calculate_max_drawdown(),
            "ready_for_live": self.is_ready_for_live(),
        }
