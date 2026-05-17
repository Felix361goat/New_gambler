import logging

logger = logging.getLogger(__name__)


def kelly_stake(probability: float, decimal_odds: float, bankroll: float, config: dict, brier_score: float = None) -> float:
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

    if brier_score is not None and brier_score > config.get("betting", {}).get("brier_score_warning_threshold", 0.20):
        calibration_factor = max(0.70, 1.0 - (brier_score - 0.20) * 5)
        fraction *= calibration_factor
        logger.warning(f"Brier score {brier_score:.4f} detected — Kelly fraction reduced to {fraction:.4f}")

    max_pct = config.get("betting", {}).get("max_stake_percent", 0.05)

    fractional_kelly = full_kelly * fraction
    max_stake = bankroll * max_pct
    stake = bankroll * fractional_kelly

    stake = round(min(stake, max_stake), 2)

    # Minimum stake: ignore dust amounts
    if stake < 0.50:
        return 0.0

    return stake


def kelly_stake_portfolio(bets: list, bankroll: float, config: dict) -> list:
    """Adjusts stakes so total portfolio exposure stays within limits."""
    max_leverage = config.get("betting", {}).get("max_portfolio_leverage", 0.25)
    total_exposure = sum(b.get("stake_recommended", 0) for b in bets) / max(bankroll, 1)

    if total_exposure > max_leverage:
        reduction_factor = max_leverage / total_exposure
        for bet in bets:
            bet["stake_recommended"] = round(bet.get("stake_recommended", 0) * reduction_factor, 2)
        logger.warning(f"Portfolio exposure {total_exposure:.1%} > {max_leverage:.1%} — stakes reduced by {reduction_factor:.2f}x")

    return bets
