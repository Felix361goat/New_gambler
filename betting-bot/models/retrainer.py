import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ModelRetrainer:
    def __init__(self, db, ensemble_model, sheets_handler=None, config: dict = None):
        self.db = db
        self.ensemble = ensemble_model
        self.sheets = sheets_handler
        self.config = config or {}
        self.min_samples = self.config.get("model", {}).get("min_samples_to_train", 50)

    def retrain_weekly(self):
        logger.info("Starting weekly model retraining...")
        start_time = datetime.now()

        training_data = self.db.get_bets_for_retraining()

        if training_data.empty or len(training_data) < self.min_samples:
            logger.warning(
                f"Insufficient training data: {len(training_data)} samples (need {self.min_samples})"
            )
            return False

        results = {}

        # Retrain XGBoost
        try:
            xgb_result = self._retrain_xgboost(training_data)
            results["xgboost"] = xgb_result
        except Exception as e:
            logger.error(f"XGBoost retraining failed: {e}")
            results["xgboost"] = {"success": False, "error": str(e)}

        # Update Poisson team strengths
        try:
            poisson_result = self._retrain_poisson(training_data)
            results["poisson"] = poisson_result
        except Exception as e:
            logger.error(f"Poisson retraining failed: {e}")
            results["poisson"] = {"success": False, "error": str(e)}

        # Update ELO ratings
        try:
            elo_result = self._update_elo(training_data)
            results["elo"] = elo_result
        except Exception as e:
            logger.error(f"ELO update failed: {e}")

        # Log to sheets
        if self.sheets:
            try:
                self.sheets.update_model_health({
                    "last_retrain": datetime.now().isoformat(),
                    "samples_used": len(training_data),
                    "duration_seconds": (datetime.now() - start_time).seconds,
                    "results": results,
                })
            except Exception as e:
                logger.error(f"Failed to log retraining to sheets: {e}")

        logger.info(f"Weekly retraining complete in {(datetime.now() - start_time).seconds}s")
        return True

    def _retrain_xgboost(self, training_data) -> dict:
        import pandas as pd
        from features.builder import FeatureBuilder

        if "won" not in training_data.columns:
            return {"success": False, "reason": "no outcome data"}

        settled = training_data.dropna(subset=["won"])
        if len(settled) < self.min_samples:
            return {"success": False, "reason": f"only {len(settled)} settled bets"}

        # Build simple feature set from bet data
        feature_cols = [c for c in settled.columns if c not in
                       ["id", "created_at", "match_date", "match_id", "home_team", "away_team",
                        "league", "market", "bookmaker_name", "status", "notes", "watchable_reason",
                        "won", "pnl_simulated", "clv_score"]]

        X = settled[feature_cols].select_dtypes(include=["number"]).fillna(0)
        y = (settled["won"] == 1).astype(int)

        if len(X) < self.min_samples:
            return {"success": False, "reason": "insufficient numeric features"}

        # Keep backup of old model
        old_model = self.ensemble.xgboost
        try:
            self.ensemble.xgboost.train(X, y)
            return {"success": True, "samples": len(X)}
        except Exception as e:
            # Rollback
            self.ensemble.xgboost = old_model
            return {"success": False, "error": str(e)}

    def _retrain_poisson(self, training_data) -> dict:
        import pandas as pd
        # Re-fit on available match data (use bet data as proxy)
        if not hasattr(self.ensemble, "poisson"):
            return {"success": False}
        try:
            self.ensemble.poisson.fit(training_data)
            return {"success": True, "samples": len(training_data)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _update_elo(self, training_data) -> dict:
        if not hasattr(self.ensemble, "elo"):
            return {"success": False}
        settled = training_data[training_data.get("status", pd.Series(["placed"])) == "placed"] if "status" in training_data.columns else training_data
        count = 0
        for _, row in training_data.iterrows():
            if row.get("home_team") and row.get("away_team"):
                try:
                    self.ensemble.elo.update_ratings(dict(row))
                    count += 1
                except Exception:
                    pass
        return {"success": True, "updated": count}
