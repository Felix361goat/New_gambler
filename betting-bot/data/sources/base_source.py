from abc import ABC, abstractmethod
from typing import Optional
import logging
import pandas as pd


class BaseSource(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def fetch(self) -> dict:
        """Fetch raw data from source."""

    @abstractmethod
    def validate(self, data: dict) -> bool:
        """Validate raw data quality."""

    @abstractmethod
    def normalize(self, data: dict) -> pd.DataFrame:
        """Transform to standard internal format."""

    def fetch_and_normalize(self) -> Optional[pd.DataFrame]:
        try:
            raw = self.fetch()
            if not self.validate(raw):
                return None
            return self.normalize(raw)
        except Exception as e:
            self.logger.error(f"{self.__class__.__name__} failed: {e}")
            return None
