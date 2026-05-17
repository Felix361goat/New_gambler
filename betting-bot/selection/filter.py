import logging
from datetime import datetime, timezone
from typing import Optional
import pytz
from selection.ev_calculator import calculate_ev, calculate_ev_with_margin, breakeven_win_rate

logger = logging.getLogger(__name__)

AUSTRIAN_TZ = pytz.timezone("Europe/Vienna")


def select_daily_bets(all_predictions: list, config: dict, week_watchable_count: int) -> list:
    """
    LAYER 0: Match-fixing exclusion — never bet on excluded leagues
    LAYER 1: Hard filter — EV >= min_ev_threshold
    LAYER 2: Sort by EV descending, take top 10
    LAYER 3: Tiebreaker by league priority (only within 0.5% EV)
    LAYER 4: Watchable flag for preferred leagues
    LAYER 5: Favorite club override (80% threshold)
    LAYER 6: Match-fixing confidence penalty for flagged (not excluded) leagues
    """
    betting_cfg = config.get("betting", {})
    leagues_cfg = config.get("leagues", {})
    clubs_cfg   = config.get("favorite_clubs", [])
    sports_cfg  = config.get("sports", {})

    min_ev       = betting_cfg.get("min_ev_threshold", 0.03)
    watchable_ev = betting_cfg.get("watchable_ev_threshold", 0.024)
    max_bets     = betting_cfg.get("max_daily_bets", 10)
    min_odds     = betting_cfg.get("min_odds", 1.50)
    safety_margin = betting_cfg.get("probability_safety_margin", 0.10)
    ev_override  = betting_cfg.get("ev_override_factor", 0.80) if isinstance(clubs_cfg, dict) else 0.80

    # Match-fixing configuration
    mf_flags    = sports_cfg.get("matchfixing_risk_flags", {})
    mf_excluded = set(mf_flags.get("exclude_leagues", []))
    mf_flagged  = set(mf_flags.get("flag_leagues", []))
    # Legacy flat list support (config.yaml match_fixing_risk_leagues)
    mf_flagged  |= set(leagues_cfg.get("match_fixing_risk_leagues", []))
    mf_penalty  = leagues_cfg.get("match_fixing_confidence_penalty", 10)

    # Handle favorite_clubs being list vs dict (as in config.yaml)
    if isinstance(clubs_cfg, dict):
        favorite_clubs = clubs_cfg.get("clubs", [])
        ev_override = clubs_cfg.get("ev_override_factor", 0.80)
    elif isinstance(clubs_cfg, list):
        favorite_clubs = clubs_cfg
        ev_override = config.get("favorite_clubs_ev_override", 0.80)
    else:
        favorite_clubs = []

    priority_leagues   = leagues_cfg.get("priority", [])
    watchable_preferred = leagues_cfg.get("watchable_preferred", [])
    watchable_max      = leagues_cfg.get("watchable_max_per_week", 3)
    kickoff_window     = leagues_cfg.get("watchable_kickoff_window", {"earliest_hour": 17, "latest_hour": 23})

    # LAYER 0: Hard exclusion — match-fixing risk too high to ever bet
    def is_excluded(pred: dict) -> bool:
        return pred.get("league", "") in mf_excluded

    # LAYER 1: Hard EV filter
    qualified = []
    for pred in all_predictions:
        # Layer 0 gate
        if is_excluded(pred):
            logger.info(
                f"Match-fixing exclusion: skipping {pred.get('home_team')} vs "
                f"{pred.get('away_team')} ({pred.get('league')})"
            )
            continue

        ev              = pred.get("ev_score", 0.0)
        home            = pred.get("home_team", "")
        away            = pred.get("away_team", "")
        bookmaker_odds  = pred.get("bookmaker_odds", 0)
        our_probability = pred.get("our_probability", 0)
        market          = pred.get("market", "")
        sport           = pred.get("sport", "")
        is_fav          = any(club in [home, away] for club in favorite_clubs)
        market_threshold = _get_market_threshold(market, sport, config)
        threshold       = market_threshold * ev_override if is_fav else market_threshold

        # Minimum odds check: low odds require very high win rates — too risky
        if bookmaker_odds < min_odds:
            continue

        # Conservative EV check: apply safety margin haircut on probability
        ev_conservative = calculate_ev_with_margin(our_probability, bookmaker_odds, safety_margin)
        if ev >= threshold and ev_conservative >= threshold:
            # Minimum edge over implied probability.
            # Use Pinnacle as the sharp-money reference when available —
            # Pinnacle runs tighter margins than soft books, so their implied
            # probability is a better proxy for the "true" market price.
            # Fall back to the soft bookmaker's odds if Pinnacle is absent.
            min_edge = betting_cfg.get("min_edge_over_implied", 0.05)
            our_prob = pred.get("our_probability", 0)
            reference_odds = pred.get("pinnacle_odds") or bookmaker_odds
            if not check_edge_over_implied(our_prob, reference_odds, min_edge):
                continue

            # Match-fixing risk check
            excluded_by_fixing, fixing_reason = check_matchfixing_risk(pred, config)
            if excluded_by_fixing:
                logger.warning(f"Bet excluded (match-fixing): {pred.get('home_team')} vs {pred.get('away_team')} — {fixing_reason}")
                continue
            if fixing_reason:
                pred["matchfixing_warning"] = fixing_reason
                logger.info(f"Match-fixing flag: {fixing_reason}")

            # Confidence tier and data completeness
            features = pred.get("features")
            if features is None:
                # No feature dict attached — completeness unknown, allow through
                completeness = 1.0
            else:
                completeness = calculate_data_completeness(features)
            pred["data_completeness"] = completeness
            pred["confidence_tier"] = calculate_confidence_tier(completeness)

            # Hard gate: too much imputed data — skip
            min_completeness = betting_cfg.get("min_data_completeness", 0.40)
            if features is not None and completeness < min_completeness:
                logger.debug(
                    f"Skipping {pred.get('home_team')} vs {pred.get('away_team')} "
                    f"— data completeness {completeness:.0%} < {min_completeness:.0%}"
                )
                continue

            pred["is_favorite_club"] = is_fav
            qualified.append(pred)

    if not qualified:
        return []

    # LAYER 2: Sort by EV descending
    qualified.sort(key=lambda x: x.get("ev_score", 0.0), reverse=True)

    # LAYER 3: Stable tiebreaker within 0.5% EV groups
    def priority_score(pred):
        league = pred.get("league", "")
        try:
            return priority_leagues.index(league)
        except ValueError:
            return len(priority_leagues)

    def sort_key(pred):
        ev = pred.get("ev_score", 0.0)
        ev_bucket = round(ev / 0.005) * 0.005  # bucket by 0.5%
        return (-ev_bucket, priority_score(pred))

    qualified.sort(key=sort_key)

    # Take top max_bets
    selected = qualified[:max_bets]

    # LAYER 4: Watchable flag
    current_week_watchable = week_watchable_count
    for pred in selected:
        league = pred.get("league", "")
        ev = pred.get("ev_score", 0.0)
        kickoff = pred.get("kickoff_time")

        pred["is_watchable"] = False
        pred["watchable_reason"] = None

        if (
            league in watchable_preferred
            and ev >= watchable_ev
            and current_week_watchable < watchable_max
        ):
            kickoff_ok = _check_kickoff_window(kickoff, kickoff_window)
            if kickoff_ok:
                pred["is_watchable"] = True
                pred["watchable_reason"] = f"Preferred league ({league}) in watchable window"
                current_week_watchable += 1

    # LAYER 5: Tag favorite clubs
    for pred in selected:
        home = pred.get("home_team", "")
        away = pred.get("away_team", "")
        if any(club in [home, away] for club in favorite_clubs):
            pred["is_favorite_club"] = True

    # LAYER 6: Apply match-fixing confidence penalty for flagged (but not excluded) leagues.
    # The bet stays eligible — we don't exclude it — but the confidence score is reduced
    # to reflect higher variance / integrity uncertainty.
    for pred in selected:
        league = pred.get("league", "")
        if league in mf_flagged:
            original = pred.get("confidence_score", 50)
            pred["confidence_score"] = max(0, original - mf_penalty)
            pred["match_fixing_flagged"] = True
            logger.info(
                f"Match-fixing flag: confidence reduced by {mf_penalty} for "
                f"{pred.get('home_team')} vs {pred.get('away_team')} ({league})"
            )
        else:
            pred["match_fixing_flagged"] = False

    return selected


def calculate_confidence_tier(data_completeness: float) -> str:
    """High / Medium / Low based on how many features were available."""
    if data_completeness >= 0.80:
        return "High"
    if data_completeness >= 0.50:
        return "Medium"
    return "Low"


def calculate_data_completeness(features: dict) -> float:
    """What fraction of expected feature groups were actually computed (not imputed).

    FeatureBuilder records which groups were available vs imputed via
    _available_groups / _imputed_groups keys.  When those keys are present we
    use them for an accurate group-level completeness score.  For sport-specific
    builders that don't produce group metadata we fall back to a simple
    non-zero-value heuristic on the numeric features only.
    """
    if not features:
        return 0.0

    available = features.get("_available_groups")
    imputed   = features.get("_imputed_groups")

    if available is not None and imputed is not None:
        total_groups = len(available) + len(imputed)
        if total_groups == 0:
            return 0.0
        return round(len(available) / total_groups, 3)

    # Fallback: count numeric keys with a real (non-zero) value, skipping
    # metadata keys prefixed with "_" and the categorical sport/surface fields.
    skip_keys = {"sport", "surface"}
    numeric_items = [
        v for k, v in features.items()
        if not k.startswith("_") and k not in skip_keys and isinstance(v, (int, float, bool))
    ]
    if not numeric_items:
        return 0.0
    non_zero = sum(1 for v in numeric_items if v not in (0, 0.0, False))
    return round(non_zero / len(numeric_items), 3)


def check_matchfixing_risk(bet: dict, config: dict) -> tuple[bool, str]:
    """
    Returns (is_flagged, reason).
    Excludes table tennis entirely.
    Flags but doesn't exclude soccer tier 4/5 and polish league.
    """
    league = bet.get("league", "")
    sport = bet.get("sport", "")
    excluded = config.get("sports", {}).get("matchfixing_excluded", [])
    flagged = config.get("sports", {}).get("matchfixing_risk_flags", {}).get("flag_leagues", [])

    if league in excluded or sport in excluded:
        return True, f"Excluded: {league} has high match-fixing risk"
    if league in flagged:
        return False, f"Warning: {league} has elevated match-fixing risk — verify independently"
    return False, ""


def check_edge_over_implied(our_prob: float, bookie_odds: float, min_edge: float = 0.05) -> bool:
    """Our prob must exceed bookie implied prob by min_edge (default 5%)."""
    if bookie_odds <= 1.0:
        return False
    implied = 1.0 / bookie_odds
    return (our_prob - implied) >= min_edge


def _get_market_threshold(market: str, sport: str, config: dict) -> float:
    """
    Return the market-specific EV threshold from config.
    Falls back to betting.min_ev_threshold if no market-specific entry exists.
    """
    thresholds = config.get("betting", {}).get("market_ev_thresholds", {})
    default = thresholds.get("default", config.get("betting", {}).get("min_ev_threshold", 0.05))

    if not market:
        return default

    market_lower = market.lower()
    sport_lower = (sport or "").lower()

    # Soccer 1x2 markets
    if market_lower in ("1x2_home", "1x2_draw", "1x2_away"):
        return thresholds.get("soccer_1x2", default)

    # Totals — sport-specific
    if market_lower in ("over_2.5", "under_2.5", "over_3.5", "under_3.5"):
        if "hockey" in sport_lower:
            return thresholds.get("hockey_total", default)
        if "soccer" in sport_lower or "football" in sport_lower:
            return thresholds.get("soccer_total", default)

    # Tennis head-to-head / moneyline
    if "tennis" in sport_lower and market_lower in ("h2h", "h2h_home", "h2h_away", "moneyline"):
        return thresholds.get("tennis_ml", default)

    # Basketball moneyline
    if "basketball" in sport_lower and market_lower in ("moneyline", "h2h", "h2h_home", "h2h_away"):
        return thresholds.get("basketball_ml", default)

    return default


def _check_kickoff_window(kickoff, window: dict) -> bool:
    """Check if kickoff is within the Austrian time window."""
    if not kickoff:
        return True  # Unknown kickoff — don't block

    if isinstance(kickoff, str):
        try:
            kickoff = datetime.fromisoformat(kickoff)
        except Exception:
            return True

    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)

    try:
        kickoff_at = kickoff.astimezone(AUSTRIAN_TZ)
        hour = kickoff_at.hour
        earliest = window.get("earliest_hour", 17)
        latest = window.get("latest_hour", 23)
        return earliest <= hour <= latest
    except Exception:
        return True
