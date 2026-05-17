import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import requests
import pandas as pd

from .base_source import BaseSource


# Supported sports slugs on The Odds API
SUPPORTED_SPORTS = [
    "soccer_epl",
    "soccer_germany_bundesliga",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "basketball_nba",
]

DEFAULT_REGIONS = "eu,uk,us"
DEFAULT_MARKETS = "h2h,totals"


class OddsAPISource(BaseSource):
    """Client for The Odds API v4 (https://api.the-odds-api.com/v4).

    Fetches live/pre-match odds for all available bookmakers and provides
    a helper to extract the best available price for a given market side.
    """

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self):
        super().__init__()
        api_key = os.environ.get("ODDS_API_KEY", "")
        if not api_key:
            self.logger.warning("ODDS_API_KEY not set – requests may fail.")
        self.api_key = api_key
        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        merged_params: Dict[str, Any] = {"apiKey": self.api_key}
        if params:
            merged_params.update(params)
        resp = self.session.get(url, params=merged_params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Public fetch methods
    # ------------------------------------------------------------------

    def fetch_odds(
        self,
        sport: str = "soccer_epl",
        region: str = DEFAULT_REGIONS,
        markets: str = DEFAULT_MARKETS,
    ) -> List[Dict]:
        """Fetch live odds for *sport*.

        Args:
            sport:   The Odds API sport key, e.g. 'soccer_epl'.
            region:  Comma-separated regions: 'eu', 'uk', 'us', 'au'.
            markets: Comma-separated markets: 'h2h', 'totals', 'spreads'.

        Returns:
            List of event dicts as returned by the API, with snapshot
            metadata added under the key '_snapshot'.
        """
        data = self._get(
            f"sports/{sport}/odds",
            params={
                "regions":     region,
                "markets":     markets,
                "oddsFormat":  "decimal",
                "dateFormat":  "iso",
            },
        )

        snapshot_ts = datetime.now(timezone.utc).isoformat()
        for event in data:
            event["_snapshot"] = {
                "fetched_at": snapshot_ts,
                "sport":      sport,
                "region":     region,
                "markets":    markets,
            }
        return data

    def get_best_odds(
        self,
        match_id: str,
        market: str,
        odds_data: List[Dict],
        outcome_name: Optional[str] = None,
    ) -> Tuple[Optional[float], Optional[str]]:
        """Return (best_decimal_odds, bookmaker_name) for a given market.

        Args:
            match_id:     The Odds API event id.
            market:       Market key, e.g. 'h2h' or 'totals'.
            odds_data:    List of event dicts from :meth:`fetch_odds`.
            outcome_name: Filter by outcome name (e.g. 'Home', 'Away', 'Draw',
                          'Over', 'Under').  If None returns the best across all
                          outcomes for the market.

        Returns:
            (best_price, bookmaker_name) or (None, None) if not found.
        """
        event = next((e for e in odds_data if e.get("id") == match_id), None)
        if event is None:
            return None, None

        best_price: Optional[float] = None
        best_bookmaker: Optional[str] = None

        for bookmaker in event.get("bookmakers", []):
            bm_key = bookmaker.get("key", "")
            for mkt in bookmaker.get("markets", []):
                if mkt.get("key") != market:
                    continue
                for outcome in mkt.get("outcomes", []):
                    if outcome_name and outcome.get("name") != outcome_name:
                        continue
                    price = outcome.get("price")
                    if price is not None and (best_price is None or price > best_price):
                        best_price     = price
                        best_bookmaker = bm_key

        return best_price, best_bookmaker

    def odds_to_snapshot_dict(self, event: Dict) -> Dict:
        """Convert a single event dict to a flat snapshot record for DB storage."""
        snapshot = event.get("_snapshot", {})
        record: Dict[str, Any] = {
            "event_id":        event.get("id"),
            "sport_key":       event.get("sport_key"),
            "commence_time":   event.get("commence_time"),
            "home_team":       event.get("home_team"),
            "away_team":       event.get("away_team"),
            "fetched_at":      snapshot.get("fetched_at"),
            "bookmakers":      [],
        }

        for bm in event.get("bookmakers", []):
            bm_record: Dict[str, Any] = {
                "key":         bm.get("key"),
                "title":       bm.get("title"),
                "last_update": bm.get("last_update"),
                "markets":     [],
            }
            for mkt in bm.get("markets", []):
                mkt_record: Dict[str, Any] = {
                    "key":      mkt.get("key"),
                    "outcomes": [
                        {"name": o.get("name"), "price": o.get("price"), "point": o.get("point")}
                        for o in mkt.get("outcomes", [])
                    ],
                }
                bm_record["markets"].append(mkt_record)
            record["bookmakers"].append(bm_record)

        return record

    # ------------------------------------------------------------------
    # BaseSource interface
    # ------------------------------------------------------------------

    def fetch(self) -> Dict:
        """Fetch odds for all supported sports."""
        all_events: List[Dict] = []
        errors: List[str] = []

        for sport in SUPPORTED_SPORTS:
            try:
                events = self.fetch_odds(sport=sport)
                all_events.extend(events)
                self.logger.info(f"Fetched {len(events)} events for {sport}")
            except Exception as exc:
                msg = f"{sport}: {exc}"
                errors.append(msg)
                self.logger.warning(f"Failed to fetch odds for {msg}")

        return {"events": all_events, "errors": errors}

    def validate(self, data: Dict) -> bool:
        if not isinstance(data, dict):
            return False
        if "events" not in data:
            self.logger.warning("Validation failed: 'events' key missing")
            return False
        return True

    def normalize(self, data: Dict) -> pd.DataFrame:
        """Flatten event + bookmaker odds into a long-format DataFrame."""
        rows = []
        for event in data.get("events", []):
            event_id      = event.get("id")
            home_team     = event.get("home_team")
            away_team     = event.get("away_team")
            commence_time = event.get("commence_time", "")
            sport_key     = event.get("sport_key")
            fetched_at    = event.get("_snapshot", {}).get("fetched_at")

            for bm in event.get("bookmakers", []):
                bm_key   = bm.get("key")
                bm_title = bm.get("title")
                for mkt in bm.get("markets", []):
                    mkt_key = mkt.get("key")
                    for outcome in mkt.get("outcomes", []):
                        rows.append({
                            "event_id":       event_id,
                            "sport":          sport_key,
                            "home_team":      home_team,
                            "away_team":      away_team,
                            "commence_time":  commence_time,
                            "bookmaker_key":  bm_key,
                            "bookmaker_name": bm_title,
                            "market":         mkt_key,
                            "outcome_name":   outcome.get("name"),
                            "price":          outcome.get("price"),
                            "point":          outcome.get("point"),
                            "fetched_at":     fetched_at,
                        })

        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
            "event_id", "sport", "home_team", "away_team", "commence_time",
            "bookmaker_key", "bookmaker_name", "market", "outcome_name",
            "price", "point", "fetched_at",
        ])
        return df
