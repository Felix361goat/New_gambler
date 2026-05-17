import logging
import os
import time
import threading
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Sports supported by the-odds-api scores endpoint.
# Keys match the 'sport' field stored on bets; values are the-odds-api sport keys.
ODDS_API_SPORTS = {
    # Tennis
    "tennis_wta":       "tennis_wta",
    "tennis_itf_women": "tennis_wta",        # best available approximation
    "tennis_itf_men":   "tennis_atp",        # only approximation available
    # Legacy generic key — kept for backward compatibility
    "tennis":           "tennis_wta",
    # Hockey
    "hockey_ahl":       "icehockey_ahl",
    "hockey_echl":      "icehockey_ahl",     # ECHL not covered; AHL as fallback
    # Legacy generic key
    "hockey":           "icehockey_ahl",
    # Basketball
    "basketball_baltic":   "basketball_euroleague",
    "basketball_romanian": "basketball_euroleague",
    # Legacy generic key
    "basketball":          "basketball_euroleague",
    # Soccer
    "soccer_england_tier4": "soccer_england",
    "soccer_scandinavia":   "soccer_sweden",   # primary Scandinavian fallback
    "soccer_poland":        "soccer_poland",
}

# Mapping from bet market to how we label the actual result
MARKET_LABEL_MAP = {
    "over_2.5":   "over",
    "under_2.5":  "under",
    "1x2_home":   "home_win",
    "1x2_draw":   "draw",
    "1x2_away":   "away_win",
    "btts":       "btts",
}


def _determine_winner(market: str, home_goals: int, away_goals: int) -> Tuple[str, bool]:
    """Return (actual_result_label, won) for a football bet given final score.

    Returns (actual_result_label, won_bool).
    """
    total = home_goals + away_goals

    if market == "over_2.5":
        won = total > 2
        return "over" if won else "under", won

    if market == "under_2.5":
        won = total <= 2
        return "under" if won else "over", won

    if market == "1x2_home":
        won = home_goals > away_goals
        if home_goals > away_goals:
            result = "home_win"
        elif home_goals == away_goals:
            result = "draw"
        else:
            result = "away_win"
        return result, won

    if market == "1x2_draw":
        won = home_goals == away_goals
        if home_goals > away_goals:
            result = "home_win"
        elif home_goals == away_goals:
            result = "draw"
        else:
            result = "away_win"
        return result, won

    if market == "1x2_away":
        won = away_goals > home_goals
        if home_goals > away_goals:
            result = "home_win"
        elif home_goals == away_goals:
            result = "draw"
        else:
            result = "away_win"
        return result, won

    if market == "btts":
        won = home_goals > 0 and away_goals > 0
        return "btts" if won else "no_btts", won

    raise ValueError(f"Unknown market: {market}")


class ResultsSource:
    """Fetches actual match outcomes and settles open bets in the database.

    Supports:
    - Football (soccer) via football-data.org API
    - Tennis, Hockey, Basketball via the-odds-api.com scores endpoint

    Unsettled statuses considered: 'placed', 'pending'.
    """

    FOOTBALL_BASE_URL = "https://api.football-data.org/v4"
    ODDS_BASE_URL = "https://api.the-odds-api.com/v4"

    # football-data.org free tier: max 10 req / 60 s
    MAX_FOOTBALL_REQUESTS_PER_MINUTE = 10

    def __init__(self):
        football_key = os.environ.get("FOOTBALL_API_KEY", "")
        odds_key = os.environ.get("ODDS_API_KEY", "")

        if not football_key:
            logger.warning("FOOTBALL_API_KEY not set — football result lookups will fail.")
        if not odds_key:
            logger.warning("ODDS_API_KEY not set — non-football result lookups will fail.")

        self._football_session = requests.Session()
        self._football_session.headers.update({
            "X-Auth-Token": football_key,
            "Content-Type": "application/json",
        })

        self._odds_api_key = odds_key

        # Rate-limiter state for football-data.org
        self._rate_lock = threading.Lock()
        self._request_timestamps: List[float] = []

    # ------------------------------------------------------------------
    # Rate limiting (football-data.org)
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        """Block until the next request is within the rate limit."""
        with self._rate_lock:
            now = time.monotonic()
            window_start = now - 60.0
            self._request_timestamps = [
                ts for ts in self._request_timestamps if ts > window_start
            ]
            if len(self._request_timestamps) >= self.MAX_FOOTBALL_REQUESTS_PER_MINUTE:
                oldest = self._request_timestamps[0]
                sleep_for = 60.0 - (now - oldest) + 0.1
                if sleep_for > 0:
                    logger.debug(f"Rate limit reached — sleeping {sleep_for:.1f}s")
                    time.sleep(sleep_for)
            self._request_timestamps.append(time.monotonic())

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------

    def _football_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        """GET from football-data.org with throttling."""
        self._throttle()
        url = f"{self.FOOTBALL_BASE_URL}/{endpoint.lstrip('/')}"
        resp = self._football_session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _odds_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """GET from the-odds-api.com."""
        url = f"{self.ODDS_BASE_URL}/{endpoint.lstrip('/')}"
        merged_params: Dict[str, Any] = {"apiKey": self._odds_api_key}
        if params:
            merged_params.update(params)
        resp = requests.get(url, params=merged_params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Closing-line value helpers
    # ------------------------------------------------------------------

    def _get_closing_odds(self, db, match_id: str, market: str) -> float:
        """Return the most recent pre-match closing odds for a given match/market.

        Queries the odds_snapshots table for the last snapshot captured before
        the match started.  We use MAX(odds_home) as a proxy for the best
        available closing price; for over/under markets we use odds_over /
        odds_under instead.

        Returns 0.0 if no snapshot exists (caller keeps clv_score = 0.0).
        """
        if db is None:
            return 0.0
        try:
            if market in ("over_2.5",):
                col = "odds_over"
            elif market in ("under_2.5",):
                col = "odds_under"
            elif market in ("1x2_away",):
                col = "odds_away"
            elif market in ("1x2_draw",):
                col = "odds_draw"
            else:
                col = "odds_home"

            with db._get_conn() as conn:
                row = conn.execute(
                    f"""
                    SELECT MAX({col}) FROM odds_snapshots
                    WHERE match_id = ? AND market = ?
                    ORDER BY captured_at DESC LIMIT 1
                    """,
                    (str(match_id), market),
                ).fetchone()
            val = row[0] if row else None
            if val is None:
                logger.debug(
                    f"No closing odds found for match_id={match_id!r} market={market!r}"
                )
                return 0.0
            return float(val)
        except Exception as exc:
            logger.warning(f"_get_closing_odds failed for {match_id}/{market}: {exc}")
            return 0.0

    # ------------------------------------------------------------------
    # Football result lookup
    # ------------------------------------------------------------------

    def _fetch_football_match(self, match_id: str) -> Optional[Dict]:
        """Fetch a single match from football-data.org by match_id.

        Returns the raw match dict, or None if not found / not finished.
        """
        try:
            data = self._football_get(f"matches/{match_id}")
            return data
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.debug(f"Match {match_id} not found on football-data.org")
            else:
                logger.warning(f"HTTP error fetching football match {match_id}: {exc}")
        except Exception as exc:
            logger.warning(f"Error fetching football match {match_id}: {exc}")
        return None

    def _resolve_football_bet(self, bet: Dict, db=None) -> Optional[Dict]:
        """Try to resolve a single football bet.

        Returns a result_dict ready for db.insert_result, or None if the
        match is not yet finished / could not be found.
        """
        from selection.ev_calculator import calculate_clv

        match_id = bet.get("match_id")
        if not match_id:
            logger.debug(f"Bet {bet['id']} has no match_id — skipping")
            return None

        raw = self._fetch_football_match(match_id)
        if raw is None:
            return None

        status = raw.get("status", "")
        if status == "POSTPONED":
            logger.info(f"Match {match_id} is POSTPONED — voiding bet {bet['id']}")
            return {"__void": True, "bet_id": bet["id"]}

        if status != "FINISHED":
            logger.debug(f"Match {match_id} status={status!r} — not finished yet")
            return None

        score = raw.get("score", {})
        full_time = score.get("fullTime", {})
        home_goals = full_time.get("home")
        away_goals = full_time.get("away")

        if home_goals is None or away_goals is None:
            logger.warning(f"Match {match_id} finished but score is null — skipping")
            return None

        market = bet.get("market", "")
        try:
            actual_result, won = _determine_winner(market, int(home_goals), int(away_goals))
        except ValueError as exc:
            logger.warning(f"Bet {bet['id']}: {exc}")
            return None

        stake = bet.get("stake_recommended") or 0.0
        odds = bet.get("bookmaker_odds") or 0.0
        pnl = stake * (odds - 1) if won else -stake

        closing_odds = self._get_closing_odds(db, match_id, market)
        if closing_odds <= 0.0:
            logger.warning(
                f"Bet {bet['id']}: no closing odds for match {match_id}/{market} — CLV set to 0.0"
            )
        clv = calculate_clv(odds, closing_odds) if closing_odds > 0.0 else 0.0

        return {
            "bet_id": bet["id"],
            "actual_result": actual_result,
            "won": won,
            "pnl_simulated": round(pnl, 4),
            "closing_line_odds": closing_odds,
            "clv_score": clv,
        }

    # ------------------------------------------------------------------
    # Non-football result lookup (the-odds-api scores)
    # ------------------------------------------------------------------

    def _fetch_odds_api_scores(self, sport_key: str, days_from: int = 1) -> List[Dict]:
        """Fetch finished scores from the-odds-api for a given sport key."""
        try:
            data = self._odds_get(
                f"sports/{sport_key}/scores/",
                params={"daysFrom": days_from, "dateFormat": "iso"},
            )
            return data if isinstance(data, list) else []
        except requests.HTTPError as exc:
            logger.warning(f"HTTP error fetching {sport_key} scores: {exc}")
        except Exception as exc:
            logger.warning(f"Error fetching {sport_key} scores: {exc}")
        return []

    def _match_bet_to_score(self, bet: Dict, scores: List[Dict]) -> Optional[Dict]:
        """Match a non-football bet to a score entry by home/away team name.

        Uses case-insensitive substring matching as bookmaker and results APIs
        may use slightly different team name representations.
        """
        home = (bet.get("home_team") or "").lower()
        away = (bet.get("away_team") or "").lower()

        for game in scores:
            if not game.get("completed"):
                continue
            teams = game.get("teams", [])
            if len(teams) < 2:
                continue
            names = {t.get("name", "").lower() for t in teams}
            if any(home in n or n in home for n in names) and \
               any(away in n or n in away for n in names):
                return game

        return None

    def _resolve_non_football_bet(self, bet: Dict, sport_key: str, scores: List[Dict], db=None) -> Optional[Dict]:
        """Try to resolve a non-football bet given pre-fetched score list.

        Supports 1x2 and over/under markets for non-football sports.
        btts is left pending as it requires sport-specific score parsing.
        """
        from selection.ev_calculator import calculate_clv

        market = bet.get("market", "")

        game = self._match_bet_to_score(bet, scores)
        if game is None:
            logger.debug(
                f"Bet {bet['id']} ({bet.get('home_team')} vs {bet.get('away_team')}) "
                "not found in scores — leaving pending"
            )
            return None

        teams = game.get("teams", [])
        if len(teams) < 2:
            return None

        # the-odds-api scores endpoint: each team dict has a 'score' field
        # Scores are strings; some sports use points, some use goals.
        try:
            scores_map = {t["name"].lower(): int(t.get("score") or 0) for t in teams}
        except (KeyError, TypeError, ValueError):
            logger.warning(f"Bet {bet['id']}: could not parse scores from {teams}")
            return None

        home_name = (bet.get("home_team") or "").lower()
        away_name = (bet.get("away_team") or "").lower()

        # Find score entries by fuzzy name match
        home_score: Optional[int] = None
        away_score: Optional[int] = None
        for name, score_val in scores_map.items():
            if home_name in name or name in home_name:
                home_score = score_val
            elif away_name in name or name in away_name:
                away_score = score_val

        if home_score is None or away_score is None:
            logger.warning(
                f"Bet {bet['id']}: could not map team names to scores — "
                f"home={home_name!r}, away={away_name!r}, scores={scores_map}"
            )
            return None

        if market == "btts":
            logger.warning(
                f"Bet {bet['id']}: market {market!r} is not reliably supported "
                "for non-football sports — leaving pending"
            )
            return None

        try:
            actual_result, won = _determine_winner(market, home_score, away_score)
        except ValueError as exc:
            logger.warning(f"Bet {bet['id']}: {exc}")
            return None

        stake = bet.get("stake_recommended") or 0.0
        odds = bet.get("bookmaker_odds") or 0.0
        pnl = stake * (odds - 1) if won else -stake

        match_id = bet.get("match_id", "")
        closing_odds = self._get_closing_odds(db, match_id, market)
        if closing_odds <= 0.0:
            logger.warning(
                f"Bet {bet['id']}: no closing odds for match {match_id}/{market} — CLV set to 0.0"
            )
        clv = calculate_clv(odds, closing_odds) if closing_odds > 0.0 else 0.0

        return {
            "bet_id": bet["id"],
            "actual_result": actual_result,
            "won": won,
            "pnl_simulated": round(pnl, 4),
            "closing_line_odds": closing_odds,
            "clv_score": clv,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_results(self, db, match_date: date) -> Dict[str, int]:
        """Fetch outcomes for all unsettled bets whose match_date <= match_date.

        Settles each bet by calling db.insert_result and db.update_bet_status.

        Returns a summary dict:
            {
                "settled": int,   # bets successfully settled
                "voided":  int,   # bets voided (postponed)
                "skipped": int,   # bets left pending (no result yet / error)
            }
        """
        # Retrieve all unsettled bets up to match_date
        try:
            with db._get_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM bets
                    WHERE status IN ('placed', 'pending')
                      AND match_date <= ?
                    """,
                    (str(match_date),),
                ).fetchall()
                pending_bets: List[Dict] = [dict(r) for r in rows]
        except Exception as exc:
            logger.error(f"fetch_results: could not load pending bets: {exc}")
            return {"settled": 0, "voided": 0, "skipped": 0}

        if not pending_bets:
            logger.info("fetch_results: no unsettled bets found")
            return {"settled": 0, "voided": 0, "skipped": 0}

        logger.info(f"fetch_results: processing {len(pending_bets)} unsettled bet(s)")

        # Group bets by sport for efficient score fetching
        football_bets: List[Dict] = []
        non_football_bets: Dict[str, List[Dict]] = {}  # sport -> bets

        for bet in pending_bets:
            sport = (bet.get("sport") or "soccer").lower()
            if sport in ("soccer", "football"):
                football_bets.append(bet)
            elif sport in ODDS_API_SPORTS:
                non_football_bets.setdefault(sport, []).append(bet)
            else:
                logger.warning(
                    f"Bet {bet['id']} has unknown sport {sport!r} — skipping"
                )

        settled = 0
        voided = 0
        skipped = 0

        # --- Football bets ---
        for bet in football_bets:
            try:
                result = self._resolve_football_bet(bet, db=db)
            except Exception as exc:
                logger.error(f"Unexpected error resolving football bet {bet['id']}: {exc}")
                skipped += 1
                continue

            if result is None:
                skipped += 1
                continue

            if result.get("__void"):
                db.update_bet_status(bet["id"], "void")
                voided += 1
                logger.info(f"Bet {bet['id']} voided (postponed)")
                continue

            db.insert_result(result)
            db.update_bet_status(bet["id"], "settled")
            outcome = "WON" if result["won"] else "LOST"
            logger.info(
                f"Bet {bet['id']} settled: {outcome} | "
                f"result={result['actual_result']} pnl={result['pnl_simulated']:+.2f}"
            )
            settled += 1

        # --- Non-football bets (grouped by sport for one API call each) ---
        for sport, sport_bets in non_football_bets.items():
            sport_key = ODDS_API_SPORTS[sport]
            scores = self._fetch_odds_api_scores(sport_key, days_from=3)

            for bet in sport_bets:
                try:
                    result = self._resolve_non_football_bet(bet, sport_key, scores, db=db)
                except Exception as exc:
                    logger.error(
                        f"Unexpected error resolving {sport} bet {bet['id']}: {exc}"
                    )
                    skipped += 1
                    continue

                if result is None:
                    skipped += 1
                    continue

                if result.get("__void"):
                    db.update_bet_status(bet["id"], "void")
                    voided += 1
                    logger.info(f"Bet {bet['id']} voided (postponed)")
                    continue

                db.insert_result(result)
                db.update_bet_status(bet["id"], "settled")
                outcome = "WON" if result["won"] else "LOST"
                logger.info(
                    f"Bet {bet['id']} settled: {outcome} | "
                    f"result={result['actual_result']} pnl={result['pnl_simulated']:+.2f}"
                )
                settled += 1

        return {"settled": settled, "voided": voided, "skipped": skipped}
