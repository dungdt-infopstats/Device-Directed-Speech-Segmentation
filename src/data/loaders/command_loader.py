"""
Command data loader for TV/device commands.
"""

import pandas as pd
from pathlib import Path
from typing import List, Union
from .json_loader import JSONDataLoader
from ...core.base_data_handler import BaseDataLoader


class CommandLoader(BaseDataLoader):
    """Loader for command data."""

    def __init__(self, commands_dir: Union[str, Path]):
        self.commands_dir = Path(commands_dir)
        self.json_loader = JSONDataLoader()

    def load(self, source: Union[str, Path] = None) -> pd.DataFrame:
        """Load commands from JSON file."""
        if source:
            file_path = Path(source)
        else:
            file_path = self.commands_dir / "commands.json"

        if not file_path.exists():
            raise FileNotFoundError(f"Commands file not found: {file_path}")

        return self.json_loader.load(file_path)

    def validate(self, data: pd.DataFrame) -> bool:
        """Validate command data structure."""
        required_columns = ['command']
        return all(col in data.columns for col in required_columns)

    def get_commands_list(self) -> List[str]:
        """Get list of all commands."""
        df = self.load()
        return df['command'].tolist()