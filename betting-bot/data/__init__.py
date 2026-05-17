"""Data layer for the betting bot."""
from .collector import DataCollector, CollectionResult
from .quality_check import QualityChecker

__all__ = [
    "DataCollector",
    "CollectionResult",
    "QualityChecker",
]
