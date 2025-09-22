"""
Base data handler abstract classes for the synthesis pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from pathlib import Path
import pandas as pd


class BaseDataLoader(ABC):
    """Abstract base class for data loading operations."""

    @abstractmethod
    def load(self, source: Union[str, Path]) -> pd.DataFrame:
        """Load data from source and return as DataFrame."""
        pass

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate loaded data format."""
        pass


class BaseDataSaver(ABC):
    """Abstract base class for data saving operations."""

    @abstractmethod
    def save(self, data: pd.DataFrame, destination: Union[str, Path]) -> bool:
        """Save DataFrame to destination."""
        pass


class BaseProcessor(ABC):
    """Abstract base class for data processing operations."""

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data and return output."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return processor configuration."""
        pass


class BasePhase(ABC):
    """Abstract base class for pipeline phases."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup()

    @abstractmethod
    def setup(self):
        """Setup phase-specific components."""
        pass

    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """Execute the phase."""
        pass

    @abstractmethod
    def cleanup(self):
        """Cleanup phase resources."""
        pass