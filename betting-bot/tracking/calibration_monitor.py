import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class CalibrationMonitor:
    """Tracks model calibration quality via Brier Score and CLV."""

    def __init__(self, db):
        self.db = db

    def get_calibration_report(self) -> dict:
        """Returns calibration status and recommendation."""
        try:
            from tracking.performance import PerformanceTracker
            tracker = PerformanceTracker(self.db)
            brier = tracker.calculate_brier_score(days=30)
        except Exception:
            brier = None

        try:
            clv_ok, clv_avg = self.db.check_clv_gate()
        except Exception:
            clv_ok, clv_avg = True, None

        recommendation = "OK"
        if brier is not None and brier > 0.20:
            recommendation = "REDUCE_KELLY — models overconfident"
        if not clv_ok:
            recommendation = "REDUCE_KELLY — negative CLV"

        return {
            "brier_score": brier,
            "clv_average": clv_avg,
            "status": recommendation,
            "last_updated": datetime.now().isoformat(),
        }
