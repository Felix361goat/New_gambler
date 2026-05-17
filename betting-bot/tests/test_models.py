import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPoissonModel:
    def setup_method(self):
        from models.poisson_model import PoissonModel
        self.model = PoissonModel()

    def _make_matches(self, n=50):
        rng = np.random.default_rng(42)
        teams = ["Arsenal", "Chelsea", "Liverpool", "ManCity", "Tottenham"]
        rows = []
        for i in range(n):
            home = rng.choice(teams)
            away = rng.choice([t for t in teams if t != home])
            hg = int(rng.poisson(1.5))
            ag = int(rng.poisson(1.1))
            rows.append({
                "home_team": home, "away_team": away,
                "home_goals": hg, "away_goals": ag,
                "status": "FINISHED",
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            })
        return pd.DataFrame(rows)

    def test_probabilities_sum_to_one(self):
        df = self._make_matches(60)
        self.model.fit(df)
        pred = self.model.predict_match("Arsenal", "Chelsea")
        total = pred["home_win_prob"] + pred["draw_prob"] + pred["away_win_prob"]
        assert abs(total - 1.0) < 1e-4, f"Probs sum to {total}, not 1.0"

    def test_over_under_complement(self):
        df = self._make_matches(60)
        self.model.fit(df)
        pred = self.model.predict_match("Arsenal", "Chelsea")
        assert abs(pred["over_25_prob"] + pred["under_25_prob"] - 1.0) < 1e-4

    def test_predict_unfitted(self):
        # Should return valid probabilities even without fitting
        pred = self.model.predict_match("TeamA", "TeamB")
        assert "home_win_prob" in pred

    def test_predicted_goals_positive(self):
        df = self._make_matches(60)
        self.model.fit(df)
        pred = self.model.predict_match("Arsenal", "Chelsea")
        assert pred["predicted_home_goals"] > 0
        assert pred["predicted_away_goals"] > 0


class TestEloModel:
    def setup_method(self):
        from models.elo_model import EloModel
        self.model = EloModel()

    def test_default_rating(self):
        assert self.model.get_rating("NewTeam") == 1500.0

    def test_probs_sum_to_one(self):
        pred = self.model.predict_match("Arsenal", "Chelsea")
        total = pred["home_win_prob"] + pred["draw_prob"] + pred["away_win_prob"]
        assert abs(total - 1.0) < 1e-4

    def test_rating_updates_after_win(self):
        initial = self.model.get_rating("Arsenal")
        self.model.update_ratings({
            "home_team": "Arsenal", "away_team": "Chelsea",
            "home_goals": 3, "away_goals": 0
        })
        assert self.model.get_rating("Arsenal") > initial

    def test_rating_updates_after_loss(self):
        initial = self.model.get_rating("Arsenal")
        self.model.update_ratings({
            "home_team": "Arsenal", "away_team": "Chelsea",
            "home_goals": 0, "away_goals": 2
        })
        assert self.model.get_rating("Arsenal") < initial


class TestEnsemble:
    def setup_method(self):
        from models.poisson_model import PoissonModel
        from models.elo_model import EloModel
        from models.xgboost_model import XGBoostModel
        from models.ensemble import EnsembleModel
        import numpy as np
        import pandas as pd

        self.config = {
            "model": {
                "ensemble_weights": {"poisson": 0.35, "xgboost": 0.40, "elo": 0.25},
                "min_models_agreeing": 2,
                "form_weights": {}
            }
        }
        poisson = PoissonModel()
        elo = EloModel()
        xgb = XGBoostModel()
        self.ensemble = EnsembleModel(poisson, xgb, elo, self.config)

    def test_confidence_in_range(self):
        pred = self.ensemble.predict("Arsenal", "Chelsea", {})
        if pred:
            assert 0 <= pred["confidence_score"] <= 100

    def test_returns_none_or_dict(self):
        result = self.ensemble.predict("Arsenal", "Chelsea", {})
        assert result is None or isinstance(result, dict)
