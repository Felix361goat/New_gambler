"""Transfermarkt scraper for injury/suspension lists and squad market values.

Uses requests + BeautifulSoup with:
- 3-second polite delay between requests
- Proper browser-like headers
- robots.txt compliance check before scraping
"""
import time
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
import pandas as pd

from .base_source import BaseSource


REQUEST_DELAY = 3  # seconds

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

BASE_URL = "https://www.transfermarkt.com"

DEFAULT_HEADERS = {
    "User-Agent":      USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer":         BASE_URL,
}


class TransfermarktSource(BaseSource):
    """Scrape injury lists and squad market values from transfermarkt.com.

    Produces two DataFrames:
        injuries_df    – one row per injured/suspended player.
        squad_values_df – one row per team with market-value proxy metrics.
    """

    def __init__(self, team_slugs: Optional[List[Tuple[str, str]]] = None):
        """
        Args:
            team_slugs: List of (team_slug, team_id) tuples, e.g.
                        [('fc-barcelona', '131'), ('real-madrid', '418')].
                        If None, uses a small default set for testing.
        """
        super().__init__()
        self.team_slugs: List[Tuple[str, str]] = team_slugs or [
            ("manchester-city", "281"),
            ("liverpool-fc", "31"),
            ("fc-barcelona", "131"),
            ("real-madrid-cf", "418"),
            ("fc-bayern-munchen", "27"),
            ("borussia-dortmund", "16"),
            ("juventus-fc", "506"),
            ("ac-milan", "5"),
        ]
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self._robots_cache: Dict[str, RobotFileParser] = {}

    # ------------------------------------------------------------------
    # robots.txt compliance
    # ------------------------------------------------------------------

    def _robots_allowed(self, url: str) -> bool:
        """Return True if USER_AGENT is allowed to fetch *url* per robots.txt."""
        parsed   = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        if base_url not in self._robots_cache:
            rp = RobotFileParser()
            robots_url = f"{base_url}/robots.txt"
            try:
                rp.set_url(robots_url)
                rp.read()
                self._robots_cache[base_url] = rp
            except Exception as exc:
                self.logger.warning(f"Could not read robots.txt from {robots_url}: {exc}")
                # Fail open – allow if we cannot read robots.txt
                return True

        return self._robots_cache[base_url].can_fetch(USER_AGENT, url)

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _get(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch URL respecting robots.txt and delay, return BeautifulSoup or None."""
        if not self._robots_allowed(url):
            self.logger.info(f"robots.txt disallows: {url}")
            return None

        try:
            resp = self.session.get(url, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as exc:
            self.logger.error(f"HTTP error for {url}: {exc}")
            return None
        finally:
            time.sleep(REQUEST_DELAY)

        return BeautifulSoup(resp.text, "html.parser")

    # ------------------------------------------------------------------
    # Injury list scraping
    # ------------------------------------------------------------------

    def _parse_date(self, text: str) -> Optional[str]:
        """Try to parse various date formats used on transfermarkt to ISO date."""
        text = text.strip()
        for fmt in ("%b %d, %Y", "%d.%m.%Y", "%Y-%m-%d", "%B %d, %Y"):
            try:
                return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None if text == "-" else text

    def fetch_injuries(self, team_slug: str, team_id: str) -> List[Dict]:
        """Scrape the injury page for a single team.

        Returns list of injury dicts.
        """
        url = f"{BASE_URL}/{team_slug}/sperrenundverletzungen/verein/{team_id}"
        self.logger.info(f"Fetching injuries: {url}")
        soup = self._get(url)
        if soup is None:
            return []

        rows: List[Dict] = []

        # The injury table has class "items" on transfermarkt
        table = soup.find("table", {"class": "items"})
        if table is None:
            self.logger.warning(f"No injury table found for {team_slug}")
            return rows

        tbody = table.find("tbody")
        if tbody is None:
            return rows

        for tr in tbody.find_all("tr", recursive=False):
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue

            # Column layout (may vary slightly):
            # 0: player shirt/img  1: name+position  2: injury type  3: since  4: until/return
            # Extract player name & position from second column
            name_td = tds[1] if len(tds) > 1 else None
            player_name = None
            position    = None
            if name_td:
                a_tag = name_td.find("a")
                player_name = a_tag.get_text(strip=True) if a_tag else name_td.get_text(strip=True)
                span = name_td.find("span")
                position = span.get_text(strip=True) if span else None

            # Injury type – 3rd column
            injury_td  = tds[2] if len(tds) > 2 else None
            injury_type = injury_td.get_text(strip=True) if injury_td else None

            # Expected return – last meaningful column
            return_td = tds[-1] if tds else None
            return_text = return_td.get_text(strip=True) if return_td else None
            expected_return = self._parse_date(return_text) if return_text else None

            rows.append({
                "team_slug":            team_slug,
                "player_name":          player_name,
                "position":             position,
                "injury_type":          injury_type,
                "expected_return_date": expected_return,
            })

        return rows

    # ------------------------------------------------------------------
    # Squad market value scraping
    # ------------------------------------------------------------------

    def fetch_squad_values(self, team_slug: str, team_id: str) -> Optional[Dict]:
        """Scrape squad market values and compute top-11 average as squad depth proxy.

        Returns a dict with team metrics or None on failure.
        """
        url = f"{BASE_URL}/{team_slug}/kader/verein/{team_id}/plus/1"
        self.logger.info(f"Fetching squad values: {url}")
        soup = self._get(url)
        if soup is None:
            return None

        # Market values are in <td class="rechts hauptlink"> tags containing €-values
        value_cells = soup.find_all("td", {"class": re.compile(r"rechts\s+hauptlink")})
        values: List[float] = []

        for cell in value_cells:
            text = cell.get_text(strip=True)
            val = self._parse_market_value(text)
            if val is not None:
                values.append(val)

        if not values:
            return None

        values_sorted = sorted(values, reverse=True)
        top11 = values_sorted[:11]
        return {
            "team_slug":          team_slug,
            "total_squad_value":  sum(values),
            "top11_avg_value":    sum(top11) / len(top11) if top11 else 0.0,
            "squad_size":         len(values),
            "max_player_value":   max(values),
            "currency":           "EUR",
        }

    def _parse_market_value(self, text: str) -> Optional[float]:
        """Parse transfermarkt market value strings like '€85.00m', '€950k'."""
        text = text.replace(",", ".").replace("\xa0", "").strip()
        m = re.search(r"([\d.]+)\s*([mk]?)", text, re.IGNORECASE)
        if not m:
            return None
        number = float(m.group(1))
        suffix = m.group(2).lower()
        if suffix == "m":
            return number * 1_000_000
        if suffix == "k":
            return number * 1_000
        return number

    # ------------------------------------------------------------------
    # BaseSource interface
    # ------------------------------------------------------------------

    def fetch(self) -> Dict:
        """Fetch injuries and squad values for all configured teams."""
        all_injuries:     List[Dict] = []
        all_squad_values: List[Dict] = []

        for team_slug, team_id in self.team_slugs:
            # Injuries
            try:
                injuries = self.fetch_injuries(team_slug, team_id)
                all_injuries.extend(injuries)
                self.logger.info(f"{team_slug}: {len(injuries)} injury entries")
            except Exception as exc:
                self.logger.warning(f"Injury fetch failed for {team_slug}: {exc}")

            # Squad values
            try:
                squad = self.fetch_squad_values(team_slug, team_id)
                if squad:
                    all_squad_values.append(squad)
                    self.logger.info(f"{team_slug}: squad value data OK")
            except Exception as exc:
                self.logger.warning(f"Squad value fetch failed for {team_slug}: {exc}")

        return {
            "injuries":     all_injuries,
            "squad_values": all_squad_values,
        }

    def validate(self, data: Dict) -> bool:
        if not isinstance(data, dict):
            return False
        if "injuries" not in data or "squad_values" not in data:
            self.logger.warning("Transfermarkt validation failed: missing keys")
            return False
        return True

    def normalize(self, data: Dict) -> pd.DataFrame:
        """Return the injuries DataFrame (primary). Use get_squad_values_df() for squad data."""
        return self.build_injuries_df(data)

    # ------------------------------------------------------------------
    # DataFrame builders (public for collector to use both)
    # ------------------------------------------------------------------

    def build_injuries_df(self, data: Dict) -> pd.DataFrame:
        """Build the injuries DataFrame from raw fetch output."""
        rows = data.get("injuries", [])
        expected_cols = [
            "team_slug", "player_name", "position",
            "injury_type", "expected_return_date",
        ]
        if not rows:
            return pd.DataFrame(columns=expected_cols)
        df = pd.DataFrame(rows)
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None
        return df[expected_cols]

    def build_squad_values_df(self, data: Dict) -> pd.DataFrame:
        """Build the squad_values DataFrame from raw fetch output."""
        rows = data.get("squad_values", [])
        expected_cols = [
            "team_slug", "total_squad_value", "top11_avg_value",
            "squad_size", "max_player_value", "currency",
        ]
        if not rows:
            return pd.DataFrame(columns=expected_cols)
        df = pd.DataFrame(rows)
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None
        return df[expected_cols]
