import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Per-sport model weights.
# Poisson = 0.0 means the model is skipped entirely for that sport.
# Tennis: Set-based outcome, no goal scoring — Poisson is inappropriate.
# Basketball: High-scoring discrete distribution doesn't fit Poisson well.
# Hockey/Soccer: Goal-based scoring, Poisson is appropriate.
SPORT_WEIGHTS = {
    "tennis":     {"poisson": 0.0,  "xgboost": 0.55, "elo": 0.45},
    "hockey":     {"poisson": 0.35, "xgboost": 0.40, "elo": 0.25},
    "basketball": {"poisson": 0.0,  "xgboost": 0.60, "elo": 0.40},
    "soccer":     {"poisson": 0.30, "xgboost": 0.45, "elo": 0.25},
}


class EnsembleModel:
    def __init__(self, poisson_model, xgboost_model, elo_model, config: dict):
        self.poisson = poisson_model
        self.xgboost = xgboost_model
        self.elo = elo_model
        # Default weights from config (used as fallback for unknown sports)
        weights = config.get("model", {}).get("ensemble_weights", {})
        self.w_poisson = weights.get("poisson", 0.35)
        self.w_xgboost = weights.get("xgboost", 0.40)
        self.w_elo = weights.get("elo", 0.25)
        self.min_models_agreeing = config.get("model", {}).get("min_models_agreeing", 2)
        self.agreement_prob_spread_max = config.get("model", {}).get("agreement_prob_spread_max", 0.15)

    def predict(
        self,
        home_team: str,
        away_team: str,
        features: dict,
        sport: str = "soccer",
        surface: Optional[str] = None,
    ) -> Optional[dict]:
        # Override weights based on sport — Poisson is disabled for tennis and basketball
        weights = SPORT_WEIGHTS.get(sport, SPORT_WEIGHTS["soccer"])
        self.w_poisson = weights["poisson"]
        self.w_xgboost = weights["xgboost"]
        self.w_elo = weights["elo"]

        # Surface is relevant for tennis ELO (clay/hard/grass separate ratings).
        # Fall back to features dict if caller didn't supply it explicitly.
        resolved_surface = surface or features.get("surface")

        predictions = []

        # Only call Poisson when its weight is non-zero (disabled for tennis/basketball)
        if self.w_poisson > 0:
            try:
                p_pred = self.poisson.predict_match(home_team, away_team)
                predictions.append(("poisson", p_pred, self.w_poisson))
            except Exception as e:
                logger.warning(f"Poisson prediction failed: {e}")

        try:
            xgb_pred = self.xgboost.predict(features)
            predictions.append(("xgboost", xgb_pred, self.w_xgboost))
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")

        try:
            elo_pred = self.elo.predict_match(home_team, away_team, sport=sport, surface=resolved_surface)
            predictions.append(("elo", elo_pred, self.w_elo))
        except Exception as e:
            logger.warning(f"ELO prediction failed: {e}")

        if len(predictions) < 2:
            logger.error("Fewer than 2 models available — cannot generate prediction")
            return None

        if not self._models_agree([p[1] for p in predictions]):
            logger.info(f"Models disagree for {home_team} vs {away_team} — skipping")
            return None

        # Weighted average
        total_weight = sum(w for _, _, w in predictions)
        result = {}

        keys = ["home_win_prob", "draw_prob", "away_win_prob", "over_25_prob", "under_25_prob",
                "over_35_prob", "under_35_prob", "btts_prob"]

        for key in keys:
            weighted_sum = sum(
                pred.get(key, 0.0) * w
                for _, pred, w in predictions
                if pred.get(key) is not None
            )
            result[key] = round(weighted_sum / total_weight, 4)

        # Feature completeness
        expected_features = 30
        actual_features = len([v for v in features.values() if v != 0])
        feature_completeness = min(1.0, actual_features / expected_features)

        # Model agreement score (0-1)
        agreement = self._agreement_score([p[1] for p in predictions])

        # Data recency (default good)
        data_recency = 1.0

        confidence_raw = (agreement * 0.50 + feature_completeness * 0.30 + data_recency * 0.20)
        confidence_score = int(max(0, min(100, confidence_raw * 100)))

        # Apply news sentiment modifier
        sentiment_mod = features.get("news_sentiment_modifier", 0.0)
        confidence_score = int(max(0, min(100, confidence_score + sentiment_mod * 100)))

        result["confidence_score"] = confidence_score
        result["models_used"] = [name for name, _, _ in predictions]
        result["model_agreement"] = round(agreement, 3)

        # Normalize 1x2 probs
        total_1x2 = result.get("home_win_prob", 0) + result.get("draw_prob", 0) + result.get("away_win_prob", 0)
        if total_1x2 > 0:
            result["home_win_prob"] = round(result["home_win_prob"] / total_1x2, 4)
            result["draw_prob"] = round(result["draw_prob"] / total_1x2, 4)
            result["away_win_prob"] = round(result["away_win_prob"] / total_1x2, 4)

        return result

    def _models_agree(self, predictions: list) -> bool:
        if len(predictions) < 2:
            return False

        def outcome_and_prob(pred):
            hw = pred.get("home_win_prob", 0.33)
            d  = pred.get("draw_prob", 0.33)
            aw = pred.get("away_win_prob", 0.34)
            label = max(["home", "draw", "away"],
                        key=lambda x: {"home": hw, "draw": d, "away": aw}[x])
            prob = {"home": hw, "draw": d, "away": aw}[label]
            return label, prob

        results = [outcome_and_prob(p) for p in predictions]
        labels = [r[0] for r in results]
        most_common = max(set(labels), key=labels.count)
        agreeing = [r for r in results if r[0] == most_common]

        # When only 2 models active, require unanimous; otherwise 2/3 majority.
        required = len(predictions) if len(predictions) < 3 else self.min_models_agreeing
        if len(agreeing) < required:
            return False

        # Probability magnitude gate: agreeing models must be within 0.15 of each
        # other.  XGBoost at 0.71 and ELO at 0.52 agree directionally but diverge
        # too widely to trust the blended probability estimate.
        probs = [r[1] for r in agreeing]
        max_spread = self.agreement_prob_spread_max
        return (max(probs) - min(probs)) <= max_spread

    def _agreement_score(self, predictions: list) -> float:
        if len(predictions) < 2:
            return 0.5

        def outcome(pred):
            hw = pred.get("home_win_prob", 0.33)
            d = pred.get("draw_prob", 0.33)
            aw = pred.get("away_win_prob", 0.34)
            return max(["home", "draw", "away"], key=lambda x: {"home": hw, "draw": d, "away": aw}[x])

        outcomes = [outcome(p) for p in predictions]
        most_common = max(set(outcomes), key=outcomes.count)
        return outcomes.count(most_common) / len(outcomes)
