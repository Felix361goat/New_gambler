import logging
from typing import Optional

logger = logging.getLogger(__name__)


def calculate_ev(our_probability: float, decimal_odds: float) -> float:
    """
    EV = (our_probability * decimal_odds) - 1
    Example: P=0.55, Odds=2.10 → (0.55 * 2.10) - 1 = +0.155 = +15.5%
    """
    if our_probability <= 0 or our_probability >= 1:
        return 0.0
    if decimal_odds <= 1.0:
        return 0.0
    return round((our_probability * decimal_odds) - 1.0, 6)


def get_platform_odds(match_id: str, market: str, odds_api_data: list, primary_bookmaker: str) -> tuple[float, str]:
    """
    Return odds from the configured primary bookmaker only.
    If the primary bookmaker has no line for this market, returns (0.0, "").
    This keeps EV calculations grounded in odds you can actually get.
    """
    if not odds_api_data:
        return 0.0, ""

    bookmaker_lower = primary_bookmaker.lower()

    for entry in odds_api_data:
        if entry.get("match_id") != match_id:
            continue
        if entry.get("market") != market:
            continue
        entry_bookmaker = (entry.get("bookmaker") or "").lower()
        if bookmaker_lower in entry_bookmaker or entry_bookmaker in bookmaker_lower:
            odds = entry.get("odds", 0.0)
            if odds and odds > 1.0:
                return float(odds), entry.get("bookmaker", primary_bookmaker)

    return 0.0, ""


def find_best_odds(match_id: str, market: str, odds_api_data: list) -> tuple[float, str]:
    """
    Line shopping: scan all bookmakers, return best odds.
    Improves ROI by 1-2% over using single bookmaker.
    Returns (best_odds, bookmaker_name).
    """
    best_odds = 0.0
    best_bookmaker = "unknown"

    if not odds_api_data:
        return best_odds, best_bookmaker

    for entry in odds_api_data:
        if entry.get("match_id") != match_id:
            continue
        if entry.get("market") != market:
            continue
        odds = entry.get("odds")
        bookmaker = entry.get("bookmaker", "unknown")
        if odds and odds > best_odds:
            best_odds = odds
            best_bookmaker = bookmaker

    return best_odds, best_bookmaker


def calculate_clv(bet_odds: float, closing_odds: float) -> float:
    """
    Closing Line Value: were our odds better than closing line?
    Positive CLV = we found value. Best model health metric.
    CLV = (bet_odds / closing_odds) - 1
    """
    if not closing_odds or closing_odds <= 1.0:
        return 0.0
    if not bet_odds or bet_odds <= 1.0:
        return 0.0
    return round((bet_odds / closing_odds) - 1.0, 6)


def implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability (no vig removed)."""
    if decimal_odds <= 1.0:
        return 1.0
    return round(1.0 / decimal_odds, 6)


def remove_vig(odds_list: list[float]) -> list[float]:
    """Remove bookmaker's margin from a set of odds for a market."""
    if not odds_list:
        return odds_list
    implied = [1.0 / o for o in odds_list if o > 1.0]
    overround = sum(implied)
    if overround <= 0:
        return odds_list
    return [round(p / overround, 6) for p in implied]


def calculate_ev_with_margin(our_probability: float, decimal_odds: float, safety_margin: float = 0.10) -> float:
    """
    EV mit Sicherheitspuffer: 10% Haircut auf unsere Wahrscheinlichkeit.
    Wenn das Modell leicht falsch liegt, bleibt EV noch positiv.
    Beispiel: P=0.55, Haircut=10% → effektive P=0.495, Odds=2.10 → EV=+3.95%
    """
    conservative_prob = our_probability * (1.0 - safety_margin)
    return calculate_ev(conservative_prob, decimal_odds)


def breakeven_win_rate(decimal_odds: float) -> float:
    """
    Minimale Win Rate um Break-Even zu erreichen.
    Beispiel: Odds 2.0 → 50%, Odds 1.5 → 66.7%, Odds 3.0 → 33.3%
    """
    if decimal_odds <= 1.0:
        return 1.0
    return round(1.0 / decimal_odds, 4)


def calculate_ev_betfair(our_probability: float, decimal_odds: float, commission: float = 0.05) -> float:
    """EV calculation accounting for Betfair's commission on winnings."""
    if our_probability <= 0 or our_probability >= 1:
        return 0.0
    if decimal_odds <= 1.0:
        return 0.0
    net_odds = 1 + (decimal_odds - 1) * (1 - commission)
    return round((our_probability * net_odds) - 1.0, 6)


def net_odds_after_commission(decimal_odds: float, commission: float = 0.05) -> float:
    """Convert decimal odds to net odds after exchange commission."""
    if decimal_odds <= 1.0:
        return decimal_odds
    return round(1 + (decimal_odds - 1) * (1 - commission), 4)


def scenario_pnl(bets: list, win_rates: list = None) -> dict:
    """
    Zeigt erwarteten P&L bei verschiedenen Win Rates.
    bets: Liste von dicts mit 'stake_recommended', 'bookmaker_odds', 'our_probability'
    Gibt dict zurück: {win_rate_pct: expected_pnl}
    """
    if win_rates is None:
        win_rates = [0.50, 0.60, 0.70, 0.80, 0.90]

    if not bets:
        return {int(wr * 100): 0.0 for wr in win_rates}

    results = {}
    for wr in win_rates:
        total_pnl = 0.0
        for bet in bets:
            stake = bet.get("stake_recommended", 0) or 0
            odds = bet.get("bookmaker_odds", 1) or 1
            # Erwarteter Gewinn bei dieser Win Rate
            # Vereinfachung: alle Wetten gleich behandelt mit der gegebenen Win Rate
            win_pnl = stake * (odds - 1)
            loss_pnl = -stake
            expected = wr * win_pnl + (1 - wr) * loss_pnl
            total_pnl += expected
        results[int(wr * 100)] = round(total_pnl, 2)

    return results
