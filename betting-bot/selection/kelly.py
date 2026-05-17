import logging

logger = logging.getLogger(__name__)


def kelly_stake(probability: float, decimal_odds: float, bankroll: float, config: dict) -> float:
    """
    Full Kelly = (bp - q) / b
    b = decimal_odds - 1
    p = our_probability
    q = 1 - p

    Apply fraction from config (0.25 = Quarter Kelly).
    Cap at max_stake_percent of bankroll.
    Returns 0 if no edge.
    """
    if probability <= 0 or probability >= 1:
        return 0.0
    if decimal_odds <= 1.0:
        return 0.0
    if bankroll <= 0:
        return 0.0

    b = decimal_odds - 1.0
    p = probability
    q = 1.0 - p

    full_kelly = (b * p - q) / b

    if full_kelly <= 0:
        return 0.0

    fraction = config.get("betting", {}).get("kelly_fraction", 0.25)
    max_pct = config.get("betting", {}).get("max_stake_percent", 0.05)

    fractional_kelly = full_kelly * fraction
    max_stake = bankroll * max_pct
    stake = bankroll * fractional_kelly

    return round(min(stake, max_stake), 2)
