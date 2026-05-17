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

        # CLV-Gate: skip retraining when model is consistently overpaying
        try:
            from tracking.performance import PerformanceTracker
            tracker = PerformanceTracker(self.db, self.config)
            clv_ok, avg_clv = tracker.check_clv_gate(self.db, min_bets=30)
            if not clv_ok:
                logger.warning(
                    f"CLV gate FAILED (avg_clv={avg_clv:.4f}) — retraining skipped"
                )
                return {"success": False, "reason": "clv_gate_failed", "avg_clv": avg_clv}
        except Exception as _clv_err:
            logger.warning(f"CLV gate check failed (non-critical, continuing): {_clv_err}")

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
        import copy
        import pandas as pd

        # Prefer rich match-feature log (real features) over flat bet columns
        try:
            feature_df = self.db.get_match_features_for_retraining()
        except Exception:
            feature_df = pd.DataFrame()

        if not feature_df.empty and "outcome" in feature_df.columns and len(feature_df) >= self.min_samples:
            settled = feature_df.dropna(subset=["outcome"])
            if len(settled) >= self.min_samples:
                meta_cols = {"match_id", "match_date", "sport", "home_team", "away_team", "outcome"}
                feature_cols = [c for c in settled.columns if c not in meta_cols]
                X = settled[feature_cols].select_dtypes(include=["number"]).fillna(0)
                y = (settled["outcome"].str.lower() == "won").astype(int)
                if len(X) >= self.min_samples:
                    # Chronological holdout: last 20% as test set
                    split = int(len(X) * 0.8)
                    X_train, X_test = X.iloc[:split], X.iloc[split:]
                    y_train, y_test = y.iloc[:split], y.iloc[split:]

                    old_model = copy.deepcopy(self.ensemble.xgboost)
                    try:
                        self.ensemble.xgboost.train(X_train, y_train)
                        new_score = self._score_model(self.ensemble.xgboost, X_test, y_test)
                        old_score = self._score_model(old_model, X_test, y_test)
                        if new_score >= old_score - 0.01:
                            return {"success": True, "samples": len(X_train),
                                    "holdout_new": round(new_score, 4),
                                    "holdout_old": round(old_score, 4),
                                    "source": "match_feature_log"}
                        else:
                            self.ensemble.xgboost = old_model
                            logger.warning(f"New model ({new_score:.4f}) worse than old ({old_score:.4f}) — rolled back")
                            return {"success": False, "reason": "holdout_gate_failed",
                                    "holdout_new": round(new_score, 4), "holdout_old": round(old_score, 4)}
                    except Exception as e:
                        self.ensemble.xgboost = old_model
                        return {"success": False, "error": str(e)}

        # Fallback: use bet data columns when feature log is empty or insufficient
        if "won" not in training_data.columns:
            return {"success": False, "reason": "no outcome data"}

        settled = training_data.dropna(subset=["won"])
        if len(settled) < self.min_samples:
            return {"success": False, "reason": f"only {len(settled)} settled bets"}

        exclude_cols = {
            "id", "created_at", "match_date", "match_id", "home_team", "away_team",
            "league", "market", "bookmaker_name", "status", "notes", "watchable_reason",
            "won", "pnl_simulated", "clv_score",
        }
        feature_cols = [c for c in settled.columns if c not in exclude_cols]
        X = settled[feature_cols].select_dtypes(include=["number"]).fillna(0)
        y = (settled["won"] == 1).astype(int)

        if len(X) < self.min_samples:
            return {"success": False, "reason": "insufficient numeric features"}

        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        import copy
        old_model = copy.deepcopy(self.ensemble.xgboost)
        try:
            self.ensemble.xgboost.train(X_train, y_train)
            new_score = self._score_model(self.ensemble.xgboost, X_test, y_test)
            old_score = self._score_model(old_model, X_test, y_test)
            if new_score >= old_score - 0.01:
                return {"success": True, "samples": len(X_train), "source": "bets_fallback",
                        "holdout_new": round(new_score, 4), "holdout_old": round(old_score, 4)}
            else:
                self.ensemble.xgboost = old_model
                logger.warning(f"New model ({new_score:.4f}) worse than old ({old_score:.4f}) — rolled back")
                return {"success": False, "reason": "holdout_gate_failed",
                        "holdout_new": round(new_score, 4), "holdout_old": round(old_score, 4)}
        except Exception as e:
            self.ensemble.xgboost = old_model
            return {"success": False, "error": str(e)}

    def _score_model(self, model, X_test, y_test) -> float:
        """Accuracy score on holdout set. Returns 0.5 if model or data unavailable."""
        try:
            if X_test.empty or len(y_test) == 0:
                return 0.5
            preds = model.predict(X_test.to_dict(orient="list"))
            if isinstance(preds, dict):
                prob_key = next((k for k in preds if "prob" in k), None)
                if prob_key:
                    import numpy as np
                    pred_labels = (np.array(list(preds[prob_key])) > 0.5).astype(int)
                    return float((pred_labels == y_test.values).mean())
            return 0.5
        except Exception:
            return 0.5

    def _retrain_poisson(self, training_data) -> dict:
        if not hasattr(self.ensemble, "poisson"):
            return {"success": False}
        # Poisson needs home_goals + away_goals per match — use match_feature_log
        try:
            match_df = self.db.get_match_features_for_retraining()
        except Exception:
            match_df = None

        if match_df is not None and not match_df.empty:
            goal_cols = {"home_goals", "away_goals", "home_team", "away_team"}
            if goal_cols.issubset(set(match_df.columns)):
                poisson_data = match_df[list(goal_cols)].dropna()
                if len(poisson_data) >= self.min_samples:
                    try:
                        self.ensemble.poisson.fit(poisson_data)
                        return {"success": True, "samples": len(poisson_data), "source": "match_feature_log"}
                    except Exception as e:
                        return {"success": False, "error": str(e)}

        return {"success": False, "reason": "no goal data available for Poisson retraining"}

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
