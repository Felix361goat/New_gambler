from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class QualityChecker:
    MIN_HISTORICAL_GAMES = 5
    MIN_ODDS = 1.01
    MAX_ODDS = 50.0
    INJURY_STALENESS_HOURS = 24

    def check_match(self, match_data: dict) -> tuple[bool, list[str]]:
        warnings = []

        home_games = match_data.get("home_team_historical_games", 0)
        away_games = match_data.get("away_team_historical_games", 0)
        if home_games < self.MIN_HISTORICAL_GAMES:
            return False, [f"Insufficient home team history: {home_games} games"]
        if away_games < self.MIN_HISTORICAL_GAMES:
            return False, [f"Insufficient away team history: {away_games} games"]

        if not match_data.get("odds_available", False):
            return False, ["No odds data available"]

        match_date = match_data.get("match_date")
        if match_date:
            if isinstance(match_date, str):
                match_date = datetime.fromisoformat(match_date)
            if match_date < datetime.now():
                return False, ["Match date is in the past"]

        injury_updated = match_data.get("injury_data_updated_at")
        if injury_updated:
            if isinstance(injury_updated, str):
                injury_updated = datetime.fromisoformat(injury_updated)
            age = datetime.now() - injury_updated
            if age > timedelta(hours=self.INJURY_STALENESS_HOURS):
                warnings.append("Injury data older than 24h")
                match_data["injury_data_stale"] = True

        if not match_data.get("weather_available", True):
            warnings.append("Weather data unavailable")
            match_data["weather_available"] = False

        if not match_data.get("xg_available", True):
            warnings.append("xG data unavailable — model runs without xG features")
            match_data["xg_available"] = False

        return True, warnings

    def check_data_freshness(self, source_name: str, last_updated: Optional[datetime]) -> bool:
        if last_updated is None:
            logger.warning(f"{source_name}: no last_updated timestamp")
            return False
        thresholds = {
            "odds": timedelta(hours=6),
            "fixtures": timedelta(hours=12),
            "standings": timedelta(hours=24),
            "injuries": timedelta(hours=24),
            "xg": timedelta(hours=48),
        }
        threshold = thresholds.get(source_name, timedelta(hours=24))
        age = datetime.now() - last_updated
        if age > threshold:
            logger.warning(f"{source_name} data is stale: {age} old (threshold: {threshold})")
            return False
        return True

    def validate_odds(self, odds_value: float) -> bool:
        if odds_value is None:
            return False
        if odds_value < self.MIN_ODDS or odds_value > self.MAX_ODDS:
            logger.warning(f"Odds value {odds_value} outside valid range [{self.MIN_ODDS}, {self.MAX_ODDS}]")
            return False
        return True
