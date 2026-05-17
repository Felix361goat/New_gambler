import logging
import pickle
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "data" / "xgboost_model.pkl"


class XGBoostModel:
    def __init__(self):
        self.model_over25 = None
        self.model_outcome = None
        self.feature_columns: list[str] = []
        self.label_encoder = LabelEncoder()
        self.fitted = False

    def train(self, features_df: pd.DataFrame, targets: pd.DataFrame):
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("xgboost not installed")
            return

        if features_df is None or features_df.empty:
            logger.warning("No training data for XGBoost")
            return

        if len(features_df) < 50:
            logger.warning(f"Only {len(features_df)} samples — need at least 50")
            return

        # CRITICAL: TimeSeriesSplit — no data leakage
        tscv = TimeSeriesSplit(n_splits=5)

        X = features_df.select_dtypes(include=[np.number]).fillna(0)
        self.feature_columns = list(X.columns)

        # Target: over_2.5
        y_over25 = targets.get("over_25") if hasattr(targets, "get") else targets
        if y_over25 is None and isinstance(targets, pd.DataFrame):
            y_over25 = targets.get("over_25", pd.Series([0] * len(X)))

        # Target: match outcome (0=away, 1=draw, 2=home)
        y_outcome = None
        if isinstance(targets, pd.DataFrame) and "outcome" in targets.columns:
            y_outcome = targets["outcome"]

        params_over25 = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }

        # Train over_25 model with TimeSeriesSplit cross-validation.
        # Each fold fits a model, evaluates log-loss on the held-out fold, and
        # logs the result. This is the only place where we can detect overfitting
        # before committing to a full-data fit.
        from sklearn.metrics import log_loss as _log_loss

        fold_scores: list[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr  = y_over25.iloc[train_idx]
            y_val = y_over25.iloc[val_idx]
            try:
                fold_model = xgb.XGBClassifier(**params_over25)
                fold_model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                val_preds = fold_model.predict_proba(X_val)[:, 1]
                score = _log_loss(y_val, val_preds)
                fold_scores.append(score)
                logger.debug(f"XGBoost CV fold {fold_idx + 1}/{tscv.n_splits}: log-loss={score:.4f}")
            except Exception as e:
                logger.warning(f"XGBoost CV fold {fold_idx + 1} failed: {e}")

        if fold_scores:
            cv_mean = float(np.mean(fold_scores))
            cv_std  = float(np.std(fold_scores))
            logger.info(
                f"XGBoost over_25 CV log-loss: {cv_mean:.4f} ± {cv_std:.4f} "
                f"(best fold: {min(fold_scores):.4f}, {len(fold_scores)}/{tscv.n_splits} folds)"
            )
        else:
            logger.warning("XGBoost over_25: all CV folds failed — proceeding with full-data fit only")

        # Final fit on all data, then calibrate probabilities with isotonic regression.
        # Uncalibrated XGBoost softmax probabilities can diverge significantly from
        # true frequencies, causing EV and Kelly calculations to be unreliable.
        try:
            from sklearn.calibration import CalibratedClassifierCV
            base_over25 = xgb.XGBClassifier(**params_over25)
            base_over25.fit(
                X, y_over25,
                eval_set=[(X.iloc[-max(1, len(X)//5):], y_over25.iloc[-max(1, len(X)//5):])],
                verbose=False,
            )
            # Wrap with calibration if enough samples; fall back to base model otherwise
            if len(X) >= 100:
                self.model_over25 = CalibratedClassifierCV(
                    base_over25, method="isotonic", cv=min(5, len(X) // 20)
                )
                self.model_over25.fit(X, y_over25)
                logger.info("XGBoost over_25: isotonic calibration applied")
            else:
                self.model_over25 = base_over25
                logger.info("XGBoost over_25: calibration skipped (< 100 samples)")
        except Exception as e:
            logger.error(f"XGBoost over_25 training failed: {e}")

        if y_outcome is not None:
            try:
                from sklearn.calibration import CalibratedClassifierCV
                base_outcome = xgb.XGBClassifier(
                    n_estimators=300, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    objective="multi:softprob", num_class=3,
                    eval_metric="mlogloss", use_label_encoder=False,
                )
                base_outcome.fit(X, y_outcome, verbose=False)
                if len(X) >= 100:
                    self.model_outcome = CalibratedClassifierCV(
                        base_outcome, method="isotonic", cv=min(5, len(X) // 20)
                    )
                    self.model_outcome.fit(X, y_outcome)
                    logger.info("XGBoost outcome: isotonic calibration applied")
                else:
                    self.model_outcome = base_outcome
            except Exception as e:
                logger.error(f"XGBoost outcome training failed: {e}")

        self.fitted = True
        self._save()
        logger.info(f"XGBoost trained on {len(X)} samples with {len(self.feature_columns)} features")

    def predict(self, features: dict) -> dict:
        if not self.fitted or self.model_over25 is None:
            return {"over_25_prob": 0.5, "home_win_prob": 0.33, "draw_prob": 0.33, "away_win_prob": 0.34}

        X = pd.DataFrame([features])
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_columns].fillna(0)

        result = {}
        try:
            over25_prob = float(self.model_over25.predict_proba(X)[0][1])
            result["over_25_prob"] = round(over25_prob, 4)
            result["under_25_prob"] = round(1 - over25_prob, 4)
        except Exception as e:
            logger.error(f"XGBoost over_25 prediction failed: {e}")
            result["over_25_prob"] = 0.5
            result["under_25_prob"] = 0.5

        if self.model_outcome is not None:
            try:
                probs = self.model_outcome.predict_proba(X)[0]
                result["away_win_prob"] = round(float(probs[0]), 4)
                result["draw_prob"] = round(float(probs[1]), 4)
                result["home_win_prob"] = round(float(probs[2]), 4)
            except Exception as e:
                logger.error(f"XGBoost outcome prediction failed: {e}")
                result.update({"home_win_prob": 0.33, "draw_prob": 0.33, "away_win_prob": 0.34})
        else:
            result.update({"home_win_prob": 0.33, "draw_prob": 0.33, "away_win_prob": 0.34})

        return result

    def _save(self):
        try:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump({"over25": self.model_over25, "outcome": self.model_outcome, "features": self.feature_columns}, f)
        except Exception as e:
            logger.error(f"XGBoost save failed: {e}")

    def load(self):
        if not MODEL_PATH.exists():
            return False
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            self.model_over25 = data.get("over25")
            self.model_outcome = data.get("outcome")
            self.feature_columns = data.get("features", [])
            self.fitted = True
            return True
        except Exception as e:
            logger.error(f"XGBoost load failed: {e}")
            return False
