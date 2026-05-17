import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from notifications.morning_briefing import format_morning_briefing
from notifications.evening_summary import format_evening_summary
from notifications.weekly_report import format_weekly_report


class TestMorningBriefing:
    def test_no_bets(self):
        msg = format_morning_briefing([], {"settled_bets": 5, "bankroll": 1000})
        assert "keine" in msg.lower() or "Heute keine" in msg

    def test_with_bets(self):
        bets = [{
            "home_team": "Arsenal", "away_team": "Chelsea",
            "league": "premier_league", "market": "over_2.5",
            "bookmaker_odds": 1.95, "bookmaker_name": "Bet365",
            "ev_score": 0.073, "confidence_score": 72,
            "stake_recommended": 12.50,
            "is_favorite_club": False, "is_watchable": False,
        }]
        perf = {"settled_bets": 10, "bankroll": 1050, "starting_bankroll": 1000}
        msg = format_morning_briefing(bets, perf)
        assert "Arsenal" in msg
        assert "Chelsea" in msg
        assert "EV" in msg
        assert "/placed" in msg or "placed" in msg.lower()

    def test_favorite_club_tag(self):
        bets = [{
            "home_team": "Arsenal", "away_team": "Chelsea",
            "league": "premier_league", "market": "over_2.5",
            "bookmaker_odds": 2.10, "bookmaker_name": "Bet365",
            "ev_score": 0.10, "confidence_score": 75,
            "stake_recommended": 15.0,
            "is_favorite_club": True, "is_watchable": False,
        }]
        perf = {"settled_bets": 5, "bankroll": 1000, "starting_bankroll": 1000}
        msg = format_morning_briefing(bets, perf)
        assert "⭐" in msg

    def test_watchable_tag(self):
        bets = [{
            "home_team": "Liverpool", "away_team": "ManCity",
            "league": "premier_league", "market": "1x2_home",
            "bookmaker_odds": 2.50, "bookmaker_name": "Pinnacle",
            "ev_score": 0.05, "confidence_score": 65,
            "stake_recommended": 8.0,
            "is_favorite_club": False, "is_watchable": True,
        }]
        perf = {"settled_bets": 15, "bankroll": 1020, "starting_bankroll": 1000}
        msg = format_morning_briefing(bets, perf)
        assert "🎯" in msg


class TestEveningSummary:
    def test_basic_summary(self):
        bets = [{
            "id": 1, "home_team": "Arsenal", "away_team": "Chelsea",
            "market": "over_2.5", "status": "placed",
            "result": {"won": True, "pnl_simulated": 15.0},
        }]
        msg = format_evening_summary(bets, {"pnl_daily": 15.0}, {"total_pnl": 15.0, "roi": 1.5})
        assert "Arsenal" in msg
        assert "✅" in msg

    def test_pending_bets(self):
        bets = [{
            "id": 1, "home_team": "Bayern", "away_team": "Dortmund",
            "market": "1x2_home", "status": "pending",
            "result": {},
        }]
        msg = format_evening_summary(bets, {"pnl_daily": 0.0}, {"total_pnl": 0.0, "roi": 0.0})
        assert "⏳" in msg


class TestWeeklyReport:
    def test_basic_report(self):
        perf = {
            "total_bets": 10, "won": 6, "win_rate": 60.0,
            "roi": 5.2, "total_pnl": 52.0, "max_drawdown": 30.0,
            "avg_confidence": 70, "settled_bets": 10,
        }
        msg = format_weekly_report(perf)
        assert "ROI" in msg
        assert "Paper Mode" in msg

    def test_too_little_data_insight(self):
        perf = {
            "total_bets": 5, "won": 3, "win_rate": 60.0,
            "roi": 3.0, "total_pnl": 15.0, "max_drawdown": 5.0,
            "avg_confidence": 65, "settled_bets": 5,
        }
        msg = format_weekly_report(perf)
        assert "wenig" in msg or "Daten" in msg

    def test_league_stats_included(self):
        perf = {
            "total_bets": 20, "won": 12, "win_rate": 60.0,
            "roi": 4.0, "total_pnl": 80.0, "max_drawdown": 20.0,
            "avg_confidence": 72, "settled_bets": 20,
        }
        league_stats = {
            "premier_league": {"roi": 8.0, "bets": 10},
            "bundesliga": {"roi": -2.0, "bets": 10},
        }
        msg = format_weekly_report(perf, league_stats)
        assert "premier_league" in msg or "Beste" in msg


class TestFilter:
    def _config(self):
        return {
            "betting": {
                "min_ev_threshold": 0.03,
                "watchable_ev_threshold": 0.024,
                "max_daily_bets": 10,
                "kelly_fraction": 0.25,
                "max_stake_percent": 0.05,
            },
            "leagues": {
                "priority": ["premier_league", "bundesliga"],
                "watchable_preferred": ["premier_league"],
                "watchable_max_per_week": 3,
                "watchable_kickoff_window": {"earliest_hour": 17, "latest_hour": 23},
            },
            "favorite_clubs": ["Arsenal", "Bristol Rovers"],
        }

    def test_ev_threshold_filters(self):
        from selection.filter import select_daily_bets
        predictions = [
            # Passes: ev=5%, odds=2.10, conservative EV=0.495*2.10-1=3.95% >= 3%
            {"home_team": "Arsenal", "away_team": "Chelsea", "league": "premier_league",
             "market": "over_2.5", "ev_score": 0.05,
             "our_probability": 0.55, "bookmaker_odds": 2.10},
            # Fails: ev=2% < 3% threshold
            {"home_team": "TeamA", "away_team": "TeamB", "league": "la_liga",
             "market": "over_2.5", "ev_score": 0.02,
             "our_probability": 0.51, "bookmaker_odds": 2.00},
        ]
        selected = select_daily_bets(predictions, self._config(), 0)
        assert len(selected) == 1
        assert selected[0]["home_team"] == "Arsenal"

    def test_max_10_bets_enforced(self):
        from selection.filter import select_daily_bets
        cfg = self._config()
        cfg["favorite_clubs"] = []
        predictions = [
            {"home_team": f"H{i}", "away_team": f"A{i}", "league": "bundesliga",
             "market": "over_2.5", "ev_score": 0.04 + i * 0.01,
             "our_probability": 0.60, "bookmaker_odds": 2.10}
            for i in range(15)
        ]
        selected = select_daily_bets(predictions, cfg, 0)
        assert len(selected) <= 10

    def test_empty_predictions(self):
        from selection.filter import select_daily_bets
        selected = select_daily_bets([], self._config(), 0)
        assert selected == []

    def test_favorite_club_80pct_threshold(self):
        from selection.filter import select_daily_bets
        predictions = [
            # Arsenal at 2.5% EV — fav threshold = 3%*80% = 2.4%, passes
            # conservative EV = 0.54*2.10-1 = 13.4% >> 2.4% ✓
            {"home_team": "Arsenal", "away_team": "Chelsea", "league": "premier_league",
             "market": "1x2_home", "ev_score": 0.025,
             "our_probability": 0.60, "bookmaker_odds": 2.10},
            # Arsenal at 2.3% EV — below 2.4% threshold, fails raw ev check
            {"home_team": "Arsenal", "away_team": "Liverpool", "league": "premier_league",
             "market": "over_2.5", "ev_score": 0.023,
             "our_probability": 0.60, "bookmaker_odds": 2.10},
        ]
        selected = select_daily_bets(predictions, self._config(), 0)
        evs = [b["ev_score"] for b in selected]
        assert 0.025 in evs
        assert 0.023 not in evs
