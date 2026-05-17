import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from selection.ev_calculator import calculate_ev, calculate_clv, remove_vig, implied_probability
from selection.kelly import kelly_stake


class TestEVCalculator:
    def test_ev_positive(self):
        ev = calculate_ev(0.55, 2.10)
        assert abs(ev - 0.155) < 1e-6, f"Expected 0.155, got {ev}"

    def test_ev_negative(self):
        ev = calculate_ev(0.40, 2.10)
        assert ev < 0

    def test_ev_zero_probability(self):
        assert calculate_ev(0.0, 2.0) == 0.0

    def test_ev_odds_too_low(self):
        assert calculate_ev(0.5, 0.9) == 0.0

    def test_ev_breakeven(self):
        ev = calculate_ev(0.5, 2.0)
        assert abs(ev) < 1e-6

    def test_clv_positive(self):
        # Bet at 2.20 odds, closing at 2.00 → positive CLV
        clv = calculate_clv(2.20, 2.00)
        assert clv > 0
        assert abs(clv - 0.10) < 1e-6

    def test_clv_negative(self):
        clv = calculate_clv(1.90, 2.10)
        assert clv < 0

    def test_clv_no_closing_odds(self):
        assert calculate_clv(2.0, 0) == 0.0

    def test_remove_vig(self):
        # 1x2 market, slight overround
        odds = [2.50, 3.20, 2.90]
        fair = remove_vig(odds)
        assert abs(sum(fair) - 1.0) < 1e-4

    def test_implied_prob(self):
        assert abs(implied_probability(2.0) - 0.5) < 1e-6


class TestKelly:
    def test_positive_edge(self):
        config = {"betting": {"kelly_fraction": 0.25, "max_stake_percent": 0.05}}
        stake = kelly_stake(0.55, 2.10, 1000.0, config)
        assert stake > 0
        assert stake <= 50.0  # max 5% of 1000

    def test_no_edge(self):
        config = {"betting": {"kelly_fraction": 0.25, "max_stake_percent": 0.05}}
        stake = kelly_stake(0.40, 2.0, 1000.0, config)
        assert stake == 0.0

    def test_cap_at_max_stake(self):
        config = {"betting": {"kelly_fraction": 1.0, "max_stake_percent": 0.05}}
        # Very high edge — should be capped
        stake = kelly_stake(0.90, 3.0, 1000.0, config)
        assert stake <= 50.0

    def test_zero_bankroll(self):
        config = {"betting": {"kelly_fraction": 0.25, "max_stake_percent": 0.05}}
        assert kelly_stake(0.6, 2.0, 0, config) == 0.0

    def test_never_negative(self):
        config = {"betting": {"kelly_fraction": 0.25, "max_stake_percent": 0.05}}
        stake = kelly_stake(0.10, 1.50, 1000.0, config)
        assert stake >= 0.0


class TestRobustness:
    def test_ev_with_margin_lower_than_raw(self):
        from selection.ev_calculator import calculate_ev_with_margin
        raw = calculate_ev(0.55, 2.10)
        conservative = calculate_ev_with_margin(0.55, 2.10, safety_margin=0.10)
        assert conservative < raw

    def test_ev_with_margin_still_positive(self):
        from selection.ev_calculator import calculate_ev_with_margin
        # P=0.60, Odds=2.10: konservativ P=0.54, EV=(0.54*2.10)-1=+13.4%
        ev = calculate_ev_with_margin(0.60, 2.10, safety_margin=0.10)
        assert ev > 0

    def test_breakeven_win_rate(self):
        from selection.ev_calculator import breakeven_win_rate
        assert abs(breakeven_win_rate(2.0) - 0.50) < 1e-4
        assert abs(breakeven_win_rate(1.5) - 0.6667) < 1e-3
        assert abs(breakeven_win_rate(3.0) - 0.3333) < 1e-3

    def test_breakeven_below_min_odds_threshold(self):
        from selection.ev_calculator import breakeven_win_rate
        # Odds unter 1.50 → braucht >66.7% Win Rate → zu unsicher
        be = breakeven_win_rate(1.40)
        assert be > 0.666

    def test_scenario_pnl_profitable_at_70pct(self):
        from selection.ev_calculator import scenario_pnl
        bets = [
            {"stake_recommended": 25.0, "bookmaker_odds": 2.10, "our_probability": 0.55},
            {"stake_recommended": 20.0, "bookmaker_odds": 1.90, "our_probability": 0.58},
        ]
        scenarios = scenario_pnl(bets)
        # Bei 70% Win Rate sollte P&L positiv sein
        assert scenarios[70] > 0, f"Expected positive P&L at 70%, got {scenarios[70]}"

    def test_scenario_pnl_empty_bets(self):
        from selection.ev_calculator import scenario_pnl
        result = scenario_pnl([])
        assert all(v == 0.0 for v in result.values())
