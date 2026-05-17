import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEvBetfair:
    def test_commission_reduces_ev(self):
        from selection.ev_calculator import calculate_ev, calculate_ev_betfair
        standard = calculate_ev(0.55, 2.10)
        betfair = calculate_ev_betfair(0.55, 2.10)
        assert betfair < standard

    def test_zero_commission_equals_standard(self):
        from selection.ev_calculator import calculate_ev, calculate_ev_betfair
        standard = calculate_ev(0.55, 2.10)
        betfair_no_fee = calculate_ev_betfair(0.55, 2.10, commission=0.0)
        assert abs(standard - betfair_no_fee) < 1e-6

    def test_net_odds_after_commission(self):
        from selection.ev_calculator import net_odds_after_commission
        # Odds 2.0, 5% commission: net = 1 + (2.0-1)*0.95 = 1.95
        assert abs(net_odds_after_commission(2.0, 0.05) - 1.95) < 1e-4

    def test_even_odds_negative_ev_with_commission(self):
        from selection.ev_calculator import calculate_ev_betfair
        # P=0.50, odds=2.0, 5% commission: EV = 0.5*1.95 - 1 = -0.025
        ev = calculate_ev_betfair(0.50, 2.0, 0.05)
        assert ev < 0

    def test_invalid_probability_returns_zero(self):
        from selection.ev_calculator import calculate_ev_betfair
        assert calculate_ev_betfair(0.0, 2.0) == 0.0
        assert calculate_ev_betfair(1.0, 2.0) == 0.0

    def test_invalid_odds_returns_zero(self):
        from selection.ev_calculator import calculate_ev_betfair
        assert calculate_ev_betfair(0.55, 0.9) == 0.0


class TestResultsSource:
    def test_importable(self):
        from data.sources.results_source import ResultsSource
        assert ResultsSource is not None

    def test_odds_api_sports_mapping_correct(self):
        from data.sources.results_source import ODDS_API_SPORTS
        # Should map to WTA, AHL — NOT ATP/NHL/NBA
        assert ODDS_API_SPORTS.get("tennis_wta") == "tennis_wta"
        assert ODDS_API_SPORTS.get("hockey_ahl") == "icehockey_ahl"
        # Legacy generic keys must not point to premium leagues
        assert ODDS_API_SPORTS.get("hockey") != "icehockey_nhl"
        assert ODDS_API_SPORTS.get("tennis") != "tennis_atp"

    def test_get_closing_odds_no_db_returns_zero(self):
        from data.sources.results_source import ResultsSource
        rs = ResultsSource()
        result = rs._get_closing_odds(None, "match123", "over_2.5")
        assert result == 0.0


class TestClvGate:
    def _make_tracker(self, clv_rows=None):
        from tracking.performance import PerformanceTracker
        db = MagicMock()
        db.get_performance_summary.return_value = {
            "settled_bets": len(clv_rows) if clv_rows else 0,
            "roi": 0, "bankroll": 1000, "total_pnl": 0,
            "win_rate": 0, "max_drawdown": 0, "avg_confidence": 0,
        }
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = clv_rows or []
        db._get_conn.return_value.__enter__ = lambda s: mock_conn
        db._get_conn.return_value.__exit__ = MagicMock(return_value=False)
        return PerformanceTracker(db, {}), db

    def test_no_data_passes(self):
        tracker, db = self._make_tracker([])
        passed, avg_clv = tracker.check_clv_gate(db)
        assert passed is True
        assert avg_clv is None

    def test_positive_clv_passes(self):
        tracker, db = self._make_tracker([(0.02,)] * 35)
        passed, avg_clv = tracker.check_clv_gate(db)
        assert passed is True
        assert avg_clv > 0

    def test_negative_clv_fails(self):
        tracker, db = self._make_tracker([(-0.03,)] * 35)
        passed, avg_clv = tracker.check_clv_gate(db)
        assert passed is False
        assert avg_clv < 0

    def test_fewer_than_30_bets_passes(self):
        tracker, db = self._make_tracker([(-0.05,)] * 10)
        passed, avg_clv = tracker.check_clv_gate(db)
        # Not enough data — should pass (safe open)
        assert passed is True


class TestTennisSource:
    def test_importable(self):
        from data.sources.tennis_source import TennisSource
        assert TennisSource is not None

    def test_fetch_itf_returns_empty(self):
        from data.sources.tennis_source import TennisSource
        ts = TennisSource({})
        result = ts.fetch_itf_upcoming()
        assert isinstance(result, list)
        assert len(result) == 0

    def test_no_api_key_returns_empty_upcoming(self):
        from data.sources.tennis_source import TennisSource
        ts = TennisSource({})  # no ODDS_API_KEY
        result = ts.fetch_upcoming_matches(days_ahead=3)
        assert isinstance(result, list)


class TestSurfaceElo:
    def test_surface_ratings_independent(self):
        from models.elo_model import EloModel
        elo = EloModel()
        elo.update_tennis_rating("Swiatek", "Gauff", "clay", player1_won=True)
        elo.update_tennis_rating("Swiatek", "Gauff", "grass", player1_won=False)
        clay_r = elo.get_tennis_rating("Swiatek", "clay")
        grass_r = elo.get_tennis_rating("Swiatek", "grass")
        # Won on clay → clay rating rises; lost on grass → grass rating drops
        assert clay_r > 1500
        assert grass_r < 1500

    def test_default_rating_for_unknown_player(self):
        from models.elo_model import EloModel
        elo = EloModel()
        assert elo.get_tennis_rating("UnknownPlayer", "hard") == 1500.0

    def test_predict_uses_surface_rating(self):
        from models.elo_model import EloModel
        elo = EloModel()
        # Inflate player1's clay rating significantly
        elo.tennis_ratings["ClayKing"] = {"clay": 1900}
        elo.tennis_ratings["ClayKing"]["hard"] = 1400
        pred_clay = elo.predict_match("ClayKing", "Opponent", sport="tennis", surface="clay")
        pred_hard = elo.predict_match("ClayKing", "Opponent", sport="tennis", surface="hard")
        assert pred_clay["home_win_prob"] > pred_hard["home_win_prob"]
