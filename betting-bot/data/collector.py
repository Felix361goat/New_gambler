"""Master data collector.

Coordinates all data sources and returns a unified dict of DataFrames.
Individual source failures are caught and logged without aborting the
entire collection run.

Usage::

    from data.collector import DataCollector

    collector = DataCollector()
    results = collector.collect_all()
    # results["football_matches"]   → pd.DataFrame
    # results["odds"]               → pd.DataFrame
    # results["xg"]                 → pd.DataFrame
    # results["injuries"]           → pd.DataFrame
    # results["squad_values"]       → pd.DataFrame
    # results["weather"]            → pd.DataFrame
    # results["nba_features"]       → pd.DataFrame
    # results["nba_games"]          → pd.DataFrame
    # results["nba_injuries"]       → pd.DataFrame
    # results["news"]               → pd.DataFrame
    # results["hockey_matches"]     → pd.DataFrame
    # results["basketball_matches"] → pd.DataFrame
    # results["tennis_matches"]     → pd.DataFrame
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from .sources.football_api import FootballAPISource
from .sources.odds_api import OddsAPISource
from .sources.understat import UnderstatSource
from .sources.transfermarkt import TransfermarktSource
from .sources.weather import WeatherSource
from .sources.nba_api import NBASource
from .sources.news_rss import NewsRSSSource
from .sources.hockey_source import HockeySource
from .sources.basketball_lower import BasketballLowerSource
from .sources.tennis_source import TennisSource


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class CollectionResult:
    """Holds the outputs of a collect_all() run."""

    def __init__(self):
        # DataFrames keyed by source name
        self.data: Dict[str, Optional[pd.DataFrame]] = {}
        # Per-source success flags
        self.success: Dict[str, bool] = {}
        # Per-source error messages
        self.errors: Dict[str, str] = {}
        # Timestamp of the collection run
        self.collected_at: datetime = datetime.now(timezone.utc)

    def set(self, key: str, df: Optional[pd.DataFrame]) -> None:
        self.data[key] = df
        self.success[key] = df is not None and not df.empty

    def set_error(self, key: str, msg: str) -> None:
        self.data[key] = None
        self.success[key] = False
        self.errors[key] = msg

    def summary(self) -> str:
        lines = [f"Collection run at {self.collected_at.isoformat()}"]
        for key in sorted(self.success):
            status = "OK" if self.success[key] else "FAILED"
            err = f" – {self.errors[key]}" if key in self.errors else ""
            rows = ""
            if self.data.get(key) is not None:
                rows = f" ({len(self.data[key])} rows)"
            lines.append(f"  [{status}] {key}{rows}{err}")
        return "\n".join(lines)

    def __getitem__(self, key: str) -> Optional[pd.DataFrame]:
        return self.data.get(key)

    def __repr__(self) -> str:
        return f"<CollectionResult {self.collected_at.isoformat()} sources={list(self.success)}>"


# ---------------------------------------------------------------------------
# Master collector
# ---------------------------------------------------------------------------

class DataCollector:
    """Orchestrate all data sources.

    Each source is wrapped in a try/except so a single source failure does
    not abort the entire pipeline.  Results are returned as a
    :class:`CollectionResult` dict-like object.
    """

    def __init__(
        self,
        config: dict = None,
        db=None,
        football_source: Optional[FootballAPISource] = None,
        odds_source: Optional[OddsAPISource] = None,
        understat_source: Optional[UnderstatSource] = None,
        transfermarkt_source: Optional[TransfermarktSource] = None,
        weather_source: Optional[WeatherSource] = None,
        nba_source: Optional[NBASource] = None,
        news_source: Optional[NewsRSSSource] = None,
        hockey_source: Optional[HockeySource] = None,
        basketball_lower_source: Optional[BasketballLowerSource] = None,
        tennis_source: Optional[TennisSource] = None,
    ):
        """Allow dependency injection for testing; otherwise create defaults."""
        self.config              = config or {}
        self.db                  = db
        self.football            = football_source            or FootballAPISource()
        self.odds                = odds_source                or OddsAPISource()
        self.understat           = understat_source           or UnderstatSource()
        self.transfermarkt       = transfermarkt_source       or TransfermarktSource()
        self.weather             = weather_source             or WeatherSource()
        self.nba                 = nba_source                 or NBASource()
        self.news                = news_source                or NewsRSSSource()
        self.hockey              = hockey_source              or HockeySource(self.config or {})
        self.basketball_lower    = basketball_lower_source    or BasketballLowerSource(self.config or {})
        self.tennis              = tennis_source              or TennisSource(self.config or {})

    # ------------------------------------------------------------------
    # Individual collection helpers
    # ------------------------------------------------------------------

    def _collect_football(self, result: CollectionResult) -> None:
        """Fetch and normalize football match data."""
        try:
            df = self.football.fetch_and_normalize()
            result.set("football_matches", df)
            logger.info(
                f"Football: {'OK' if df is not None else 'empty'} "
                f"({len(df) if df is not None else 0} rows)"
            )
        except Exception as exc:
            msg = str(exc)
            result.set_error("football_matches", msg)
            logger.error(f"Football source failed: {msg}")

    def _collect_odds(self, result: CollectionResult) -> None:
        """Fetch and normalize odds data."""
        try:
            df = self.odds.fetch_and_normalize()
            result.set("odds", df)
            logger.info(
                f"Odds: {'OK' if df is not None else 'empty'} "
                f"({len(df) if df is not None else 0} rows)"
            )
        except Exception as exc:
            msg = str(exc)
            result.set_error("odds", msg)
            logger.error(f"Odds source failed: {msg}")

    def _collect_understat(self, result: CollectionResult) -> None:
        """Fetch and normalize xG data from Understat."""
        try:
            df = self.understat.fetch_and_normalize()
            result.set("xg", df)
            logger.info(
                f"Understat xG: {'OK' if df is not None else 'empty'} "
                f"({len(df) if df is not None else 0} rows)"
            )
        except Exception as exc:
            msg = str(exc)
            result.set_error("xg", msg)
            logger.error(f"Understat source failed: {msg}")

    def _collect_transfermarkt(self, result: CollectionResult) -> None:
        """Fetch injuries and squad values from Transfermarkt."""
        try:
            raw = self.transfermarkt.fetch()
            if not self.transfermarkt.validate(raw):
                result.set_error("injuries", "Transfermarkt validation failed")
                result.set_error("squad_values", "Transfermarkt validation failed")
                return

            injuries_df     = self.transfermarkt.build_injuries_df(raw)
            squad_values_df = self.transfermarkt.build_squad_values_df(raw)

            result.set("injuries", injuries_df)
            result.set("squad_values", squad_values_df)
            logger.info(
                f"Transfermarkt: injuries={len(injuries_df)} rows, "
                f"squad_values={len(squad_values_df)} rows"
            )
        except Exception as exc:
            msg = str(exc)
            result.set_error("injuries", msg)
            result.set_error("squad_values", msg)
            logger.error(f"Transfermarkt source failed: {msg}")

    def _collect_weather(
        self,
        result: CollectionResult,
        matches: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Fetch weather data for upcoming matches."""
        try:
            # If caller provided match list, update source's matches
            if matches:
                self.weather.matches = matches

            df = self.weather.fetch_and_normalize()
            result.set("weather", df)
            logger.info(
                f"Weather: {'OK' if df is not None else 'empty'} "
                f"({len(df) if df is not None else 0} rows)"
            )
        except Exception as exc:
            msg = str(exc)
            result.set_error("weather", msg)
            logger.error(f"Weather source failed: {msg}")

    def _collect_nba(self, result: CollectionResult) -> None:
        """Fetch NBA team features, game schedule, and injury data."""
        try:
            raw = self.nba.fetch()
            if not self.nba.validate(raw):
                result.set_error("nba_features", "NBA validation failed")
                return

            features_df  = self.nba.normalize(raw)
            games_df     = self.nba.get_games_df(raw)
            injuries_df  = self.nba.get_injuries_df(raw)

            result.set("nba_features",  features_df)
            result.set("nba_games",     games_df)
            result.set("nba_injuries",  injuries_df)
            logger.info(
                f"NBA: features={len(features_df)}, "
                f"games={len(games_df)}, "
                f"injuries={len(injuries_df)} rows"
            )
        except Exception as exc:
            msg = str(exc)
            result.set_error("nba_features",  msg)
            result.set_error("nba_games",     msg)
            result.set_error("nba_injuries",  msg)
            logger.error(f"NBA source failed: {msg}")

    def _collect_hockey(self, result: CollectionResult) -> None:
        """Fetch and normalize hockey match data."""
        try:
            df = self.hockey.fetch_and_normalize()
            result.set("hockey_matches", df)
            logger.info(
                f"Hockey: {'OK' if df is not None else 'empty'} "
                f"({len(df) if df is not None else 0} rows)"
            )
        except Exception as exc:
            msg = str(exc)
            result.set_error("hockey_matches", msg)
            logger.error(f"Hockey source failed: {msg}")

    def _collect_basketball_lower(self, result: CollectionResult) -> None:
        """Fetch and normalize lower-tier European basketball data."""
        try:
            df = self.basketball_lower.fetch_and_normalize()
            result.set("basketball_matches", df)
            logger.info(
                f"BasketballLower: {'OK' if df is not None else 'empty'} "
                f"({len(df) if df is not None else 0} rows)"
            )
        except Exception as exc:
            msg = str(exc)
            result.set_error("basketball_matches", msg)
            logger.error(f"BasketballLower source failed: {msg}")

    def _collect_tennis(self, result: CollectionResult) -> None:
        """Fetch and normalize tennis match data (WTA/ITF)."""
        try:
            df = self.tennis.fetch_and_normalize()
            result.set("tennis_matches", df)
            logger.info(
                f"Tennis: {'OK' if df is not None else 'empty'} "
                f"({len(df) if df is not None else 0} rows)"
            )
        except Exception as exc:
            msg = str(exc)
            result.set_error("tennis_matches", msg)
            logger.error(f"Tennis source failed: {msg}")

    def _collect_news(self, result: CollectionResult) -> None:
        """Fetch and normalise news RSS feed entries."""
        try:
            df = self.news.fetch_and_normalize()
            result.set("news", df)
            logger.info(
                f"News RSS: {'OK' if df is not None else 'empty'} "
                f"({len(df) if df is not None else 0} rows)"
            )
        except Exception as exc:
            msg = str(exc)
            result.set_error("news", msg)
            logger.error(f"News RSS source failed: {msg}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def collect_all(
        self,
        weather_matches: Optional[List[Dict[str, Any]]] = None,
    ) -> CollectionResult:
        """Run all data sources and return a unified CollectionResult.

        Args:
            weather_matches: Optional list of dicts with 'city', 'kickoff_utc',
                             and optionally 'match_id'.  If provided, weather
                             is fetched for these matches.

        Returns:
            :class:`CollectionResult` with one DataFrame per source key.
            Sources that fail do NOT raise – they are logged and their entries
            in the result have ``success=False``.
        """
        logger.info("=== DataCollector: starting collect_all() ===")
        result = CollectionResult()

        self._collect_football(result)
        self._collect_odds(result)
        self._collect_understat(result)
        self._collect_transfermarkt(result)
        self._collect_weather(result, weather_matches)
        self._collect_nba(result)
        self._collect_news(result)
        self._collect_hockey(result)
        self._collect_basketball_lower(result)
        self._collect_tennis(result)

        # Log summary
        n_ok     = sum(1 for v in result.success.values() if v)
        n_total  = len(result.success)
        logger.info(f"=== collect_all() done: {n_ok}/{n_total} sources OK ===")
        logger.info(result.summary())

        return result

    def collect_football_only(self) -> CollectionResult:
        """Convenience method – only fetch football data (fast path)."""
        result = CollectionResult()
        self._collect_football(result)
        self._collect_odds(result)
        return result

    def collect_nba_only(self) -> CollectionResult:
        """Convenience method – only fetch NBA data."""
        result = CollectionResult()
        self._collect_nba(result)
        return result
