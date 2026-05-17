"""OpenWeatherMap forecast source.

Fetches weather conditions (wind, precipitation, temperature) for a stadium
city at the time of kick-off.  Results are cached in-process using a dict
keyed by ``f"{city}_{timestamp_hour}"`` so repeated lookups within the same
hour are free.

Environment variable:
    WEATHER_API_KEY – OpenWeatherMap API key (required for live data).
"""
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

from .base_source import BaseSource


class WeatherSource(BaseSource):
    """Fetch forecast weather for a list of (city, kick_off_utc) pairs.

    Usage::

        source = WeatherSource()
        wx = source.get_match_weather("London", "2024-03-15T15:00:00Z")
        # {"wind_speed_kmh": 18.7, "precipitation_mm": 0.0, "temperature_c": 12.3}
    """

    BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"
    # OWM free forecast gives 5-day / 3-hour intervals – max ~40 entries.
    _SECONDS_PER_HOUR = 3600

    def __init__(self, matches: Optional[List[Dict]] = None):
        """
        Args:
            matches: Optional list of dicts with keys 'city' and 'kickoff_utc'
                     (ISO-8601 string).  Used by fetch() / fetch_and_normalize().
        """
        super().__init__()
        self.api_key = os.environ.get("WEATHER_API_KEY", "")
        if not self.api_key:
            self.logger.warning("WEATHER_API_KEY not set – weather lookups will fail.")
        self.matches: List[Dict] = matches or []
        self.session = requests.Session()
        # In-process cache: key → weather dict
        self._cache: Dict[str, Optional[Dict]] = {}

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(city: str, kickoff_utc: str) -> str:
        """Build cache key from city and truncated hour string."""
        # Normalise ISO timestamp to the nearest hour
        try:
            dt = datetime.fromisoformat(kickoff_utc.replace("Z", "+00:00"))
            hour_str = dt.strftime("%Y%m%d%H")
        except Exception:
            hour_str = kickoff_utc[:13].replace("T", "").replace("-", "").replace(":", "")
        return f"{city.lower().replace(' ', '_')}_{hour_str}"

    # ------------------------------------------------------------------
    # Internal HTTP
    # ------------------------------------------------------------------

    def _fetch_forecast(self, city: str) -> Optional[List[Dict]]:
        """Return the raw OWM forecast list for *city*, or None on error."""
        if not self.api_key:
            return None
        try:
            resp = self.session.get(
                self.BASE_URL,
                params={
                    "q":     city,
                    "appid": self.api_key,
                    "units": "metric",
                    "cnt":   40,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("list", [])
        except requests.HTTPError as exc:
            self.logger.error(f"OWM HTTP error for city '{city}': {exc}")
            return None
        except Exception as exc:
            self.logger.error(f"Weather fetch failed for city '{city}': {exc}")
            return None

    @staticmethod
    def _closest_forecast(
        forecast_list: List[Dict], target_ts: float
    ) -> Optional[Dict]:
        """Return the forecast entry whose dt is closest to *target_ts* (Unix)."""
        if not forecast_list:
            return None
        return min(forecast_list, key=lambda e: abs(e.get("dt", 0) - target_ts))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_match_weather(
        self, city: str, kickoff_utc: str
    ) -> Optional[Dict]:
        """Return weather conditions for *city* at *kickoff_utc*.

        Returns a dict with keys:
            wind_speed_kmh   – wind speed in km/h
            precipitation_mm – 3-hour precipitation accumulation in mm
            temperature_c    – temperature in Celsius

        Returns None if the API fails or the key is missing.
        """
        key = self._cache_key(city, kickoff_utc)
        if key in self._cache:
            return self._cache[key]

        forecast_list = self._fetch_forecast(city)
        if forecast_list is None:
            self._cache[key] = None
            return None

        # Convert kickoff to Unix timestamp for closest-match lookup
        try:
            dt = datetime.fromisoformat(kickoff_utc.replace("Z", "+00:00"))
            target_ts = dt.timestamp()
        except Exception:
            self.logger.warning(f"Could not parse kickoff time: {kickoff_utc}")
            self._cache[key] = None
            return None

        entry = self._closest_forecast(forecast_list, target_ts)
        if entry is None:
            self._cache[key] = None
            return None

        # Wind: OWM gives m/s – convert to km/h
        wind_mps = entry.get("wind", {}).get("speed", 0.0)
        wind_kmh = round(wind_mps * 3.6, 2)

        # Precipitation: sum rain + snow for the 3-hour window
        rain_mm = entry.get("rain", {}).get("3h", 0.0)
        snow_mm = entry.get("snow", {}).get("3h", 0.0)
        precip_mm = round(rain_mm + snow_mm, 2)

        # Temperature
        temp_c = entry.get("main", {}).get("temp", None)

        result = {
            "wind_speed_kmh":   wind_kmh,
            "precipitation_mm": precip_mm,
            "temperature_c":    temp_c,
            "weather_desc":     entry.get("weather", [{}])[0].get("description", ""),
            "forecast_dt":      entry.get("dt"),
            "city":             city,
        }

        self._cache[key] = result
        return result

    def get_weather_for_matches(self, matches: List[Dict]) -> List[Optional[Dict]]:
        """Batch-fetch weather for a list of match dicts.

        Each match dict must contain:
            city        – stadium city name
            kickoff_utc – ISO-8601 UTC string

        Returns list of weather dicts (or None entries on failure) in the
        same order as *matches*.
        """
        results = []
        for match in matches:
            city       = match.get("city", "")
            kickoff    = match.get("kickoff_utc", "")
            if not city or not kickoff:
                results.append(None)
                continue
            wx = self.get_match_weather(city, kickoff)
            results.append(wx)
        return results

    # ------------------------------------------------------------------
    # BaseSource interface
    # ------------------------------------------------------------------

    def fetch(self) -> Dict:
        """Fetch weather for all matches provided at construction time."""
        results = []
        for match in self.matches:
            city    = match.get("city", "")
            kickoff = match.get("kickoff_utc", "")
            wx = self.get_match_weather(city, kickoff) if city and kickoff else None
            results.append({
                "match_id":   match.get("match_id"),
                "city":       city,
                "kickoff_utc": kickoff,
                "weather":    wx,
            })
        return {"weather_results": results}

    def validate(self, data: Dict) -> bool:
        return isinstance(data, dict) and "weather_results" in data

    def normalize(self, data: Dict) -> "pd.DataFrame":
        import pandas as pd

        rows = []
        for item in data.get("weather_results", []):
            wx = item.get("weather") or {}
            rows.append({
                "match_id":         item.get("match_id"),
                "city":             item.get("city"),
                "kickoff_utc":      item.get("kickoff_utc"),
                "wind_speed_kmh":   wx.get("wind_speed_kmh"),
                "precipitation_mm": wx.get("precipitation_mm"),
                "temperature_c":    wx.get("temperature_c"),
                "weather_desc":     wx.get("weather_desc"),
            })

        expected_cols = [
            "match_id", "city", "kickoff_utc",
            "wind_speed_kmh", "precipitation_mm", "temperature_c", "weather_desc",
        ]
        if not rows:
            return pd.DataFrame(columns=expected_cols)

        df = pd.DataFrame(rows)
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None
        return df[expected_cols]
