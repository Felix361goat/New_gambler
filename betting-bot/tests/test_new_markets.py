import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTennisFeatures:
    def _make_tennis_df(self):
        return pd.DataFrame([
            {"player1": "Swiatek", "player2": "Sabalenka", "surface": "clay",
             "winner": "Swiatek", "date": pd.Timestamp("2024-01-15")},
            {"player1": "Swiatek", "player2": "Keys", "surface": "clay",
             "winner": "Swiatek", "date": pd.Timestamp("2024-02-01")},
            {"player1": "Rybakina", "player2": "Swiatek", "surface": "hard",
             "winner": "Rybakina", "date": pd.Timestamp("2024-03-01")},
            {"player1": "Swiatek", "player2": "Gauff", "surface": "hard",
             "winner": "Swiatek", "date": pd.Timestamp("2024-04-01")},
            {"player1": "Swiatek", "player2": "Pegula", "surface": "clay",
             "winner": "Swiatek", "date": pd.Timestamp("2024-05-01")},
        ])

    def test_surface_win_rate(self):
        from features.tennis_features import calculate_tennis_features
        df = self._make_tennis_df()
        features = calculate_tennis_features("Swiatek", "Sabalenka", "clay", df, {})
        assert "surface_clay_win_rate_24m" in features
        assert features["surface_clay_win_rate_24m"] > 0

    def test_empty_data_returns_defaults(self):
        from features.tennis_features import calculate_tennis_features
        features = calculate_tennis_features("Unknown", "Player", "hard", None, {})
        assert features["sport"] == "tennis"
        assert "form_last5_all" in features

    def test_fatigue_flag(self):
        from features.tennis_features import _tournament_fatigue
        # 8 matches in 14 days = fatigue
        dates = [pd.Timestamp.now() - pd.Timedelta(days=i) for i in range(8)]
        df = pd.DataFrame([
            {"player1": "Player", "player2": f"Opp{i}", "date": d, "winner": "Player"}
            for i, d in enumerate(dates)
        ])
        result = _tournament_fatigue("Player", df)
        assert result["fatigue_flag"] is True

    def test_h2h_surface(self):
        from features.tennis_features import _h2h_surface
        df = self._make_tennis_df()
        result = _h2h_surface("Swiatek", "Sabalenka", "clay", df)
        assert result["h2h_surface_matches"] >= 1
        assert result["h2h_surface_win_rate"] == 1.0  # Swiatek won the one clay match


class TestHockeyFeatures:
    def _make_hockey_df(self):
        now = pd.Timestamp.now()
        return pd.DataFrame([
            {"home_team": "TeamA", "away_team": "TeamB", "date": now - pd.Timedelta(days=1),
             "home_goals": 3, "away_goals": 2},
            {"home_team": "TeamC", "away_team": "TeamA", "date": now - pd.Timedelta(days=3),
             "home_goals": 1, "away_goals": 2},
        ])

    def test_back_to_back_detected(self):
        from features.hockey_features import _back_to_back_features
        df = self._make_hockey_df()
        now = pd.Timestamp.now()
        result = _back_to_back_features("TeamA", now, df, "home")
        assert result["home_back_to_back"] is True
        assert result["home_b2b_penalty"] == 0.07

    def test_no_back_to_back(self):
        from features.hockey_features import _back_to_back_features
        df = self._make_hockey_df()
        now = pd.Timestamp.now()
        result = _back_to_back_features("TeamB", now, df, "away")
        # TeamB's last game was yesterday — but as away team
        assert result["away_days_rest"] <= 2

    def test_empty_data(self):
        from features.hockey_features import calculate_hockey_features
        features = calculate_hockey_features("TeamA", "TeamB", pd.Timestamp.now(), None, {})
        assert features["sport"] == "hockey"
        assert "home_back_to_back" in features


class TestBasketballFeatures:
    def test_player_impact_no_injuries(self):
        from features.basketball_lower import _player_impact
        result = _player_impact("TeamA", [], {}, "home")
        assert result["home_player_impact_score"] == 1.0
        assert result["home_key_player_missing"] is False

    def test_player_impact_key_missing(self):
        from features.basketball_lower import _player_impact
        injuries = [{"team": "TeamA", "player_name": "StarPlayer"}]
        roster = {"TeamA": [{"name": "StarPlayer"}, {"name": "Player2"}, {"name": "Player3"}]}
        result = _player_impact("TeamA", injuries, roster, "home")
        assert result["home_key_player_missing"] is True
        assert result["home_player_impact_score"] < 1.0


class TestMatchFixingFilter:
    def test_table_tennis_excluded(self):
        from selection.filter import check_matchfixing_risk
        config = {"sports": {"matchfixing_excluded": ["table_tennis"], "matchfixing_risk_flags": {"flag_leagues": []}}}
        excluded, reason = check_matchfixing_risk({"league": "table_tennis"}, config)
        assert excluded is True

    def test_polish_league_flagged_not_excluded(self):
        from selection.filter import check_matchfixing_risk
        config = {"sports": {"matchfixing_excluded": ["table_tennis"], "matchfixing_risk_flags": {"flag_leagues": ["soccer_poland"]}}}
        excluded, reason = check_matchfixing_risk({"league": "soccer_poland"}, config)
        assert excluded is False
        assert "soccer_poland" in reason


class TestEdgeCheck:
    def test_sufficient_edge(self):
        from selection.filter import check_edge_over_implied
        # Bookie odds 2.20 → implied 45.5%, our prob 52% → edge = 6.5% > 5%
        assert check_edge_over_implied(0.52, 2.20, 0.05) is True

    def test_insufficient_edge(self):
        from selection.filter import check_edge_over_implied
        # Bookie odds 2.00 → implied 50%, our prob 53% → edge = 3% < 5%
        assert check_edge_over_implied(0.53, 2.00, 0.05) is False

    def test_confidence_tier(self):
        from selection.filter import calculate_confidence_tier
        assert calculate_confidence_tier(0.85) == "High"
        assert calculate_confidence_tier(0.65) == "Medium"
        assert calculate_confidence_tier(0.30) == "Low"
