import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.quality_check import QualityChecker


class TestQualityChecker:
    def setup_method(self):
        self.qc = QualityChecker()

    def test_valid_match_passes(self):
        match_data = {
            "home_team_historical_games": 10,
            "away_team_historical_games": 8,
            "odds_available": True,
            "match_date": datetime.now() + timedelta(days=2),
        }
        valid, warnings = self.qc.check_match(match_data)
        assert valid is True

    def test_insufficient_history_fails(self):
        match_data = {
            "home_team_historical_games": 3,
            "away_team_historical_games": 8,
            "odds_available": True,
            "match_date": datetime.now() + timedelta(days=2),
        }
        valid, warnings = self.qc.check_match(match_data)
        assert valid is False

    def test_no_odds_fails(self):
        match_data = {
            "home_team_historical_games": 10,
            "away_team_historical_games": 8,
            "odds_available": False,
            "match_date": datetime.now() + timedelta(days=2),
        }
        valid, warnings = self.qc.check_match(match_data)
        assert valid is False

    def test_past_match_fails(self):
        match_data = {
            "home_team_historical_games": 10,
            "away_team_historical_games": 8,
            "odds_available": True,
            "match_date": datetime.now() - timedelta(days=1),
        }
        valid, warnings = self.qc.check_match(match_data)
        assert valid is False

    def test_stale_injury_data_warns(self):
        match_data = {
            "home_team_historical_games": 10,
            "away_team_historical_games": 8,
            "odds_available": True,
            "match_date": datetime.now() + timedelta(days=2),
            "injury_data_updated_at": datetime.now() - timedelta(hours=30),
        }
        valid, warnings = self.qc.check_match(match_data)
        assert valid is True
        assert any("24h" in w for w in warnings)
        assert match_data.get("injury_data_stale") is True

    def test_validate_odds_valid(self):
        assert self.qc.validate_odds(2.5) is True
        assert self.qc.validate_odds(1.01) is True
        assert self.qc.validate_odds(50.0) is True

    def test_validate_odds_invalid(self):
        assert self.qc.validate_odds(1.005) is False
        assert self.qc.validate_odds(51.0) is False
        assert self.qc.validate_odds(None) is False
        assert self.qc.validate_odds(0.5) is False


class TestDatabase:
    def setup_method(self):
        import tempfile
        from tracking.database import DatabaseHandler
        self.tmp = tempfile.mkdtemp()
        db_path = Path(self.tmp) / "test.db"
        self.db = DatabaseHandler(db_path=db_path)

    def test_insert_and_retrieve_bet(self):
        from datetime import date
        bet = {
            "match_date": str(date.today()),
            "league": "premier_league",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "market": "over_2.5",
            "our_probability": 0.55,
            "bookmaker_odds": 1.95,
            "bookmaker_name": "Bet365",
            "ev_score": 0.0725,
        }
        bet_id = self.db.insert_bet(bet)
        assert bet_id is not None
        assert bet_id > 0

        pending = self.db.get_pending_bets(date.today())
        assert len(pending) == 1
        assert pending[0]["home_team"] == "Arsenal"

    def test_update_bet_status(self):
        from datetime import date
        bet = {
            "match_date": str(date.today()),
            "league": "bundesliga",
            "home_team": "Bayern",
            "away_team": "Dortmund",
            "market": "1x2_home",
            "our_probability": 0.60,
            "bookmaker_odds": 1.75,
            "bookmaker_name": "Pinnacle",
            "ev_score": 0.05,
        }
        bet_id = self.db.insert_bet(bet)
        success = self.db.update_bet_status(bet_id, "placed")
        assert success is True

        with self.db._get_conn() as conn:
            row = conn.execute("SELECT status FROM bets WHERE id = ?", (bet_id,)).fetchone()
            assert row[0] == "placed"
