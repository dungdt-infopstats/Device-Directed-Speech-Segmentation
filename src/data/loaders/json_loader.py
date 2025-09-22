"""
JSON data loader for the synthesis pipeline.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from ..loaders import BaseDataLoader


class JSONDataLoader(BaseDataLoader):
    """JSON data loader that converts JSON arrays to DataFrames."""

    def load(self, source: Union[str, Path]) -> pd.DataFrame:
        """Load JSON file and convert to DataFrame."""
        with open(source, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict) and 'data' in data:
            # Handle structured JSON with metadata
            df = pd.DataFrame(data['data'])
            if 'metadata' in data:
                df.attrs = data['metadata']
            return df
        else:
            raise ValueError(f"Unsupported JSON format in {source}")

    def validate(self, data: pd.DataFrame) -> bool:
        """Validate DataFrame structure."""
        return isinstance(data, pd.DataFrame) and not data.empty

    def load_multiple_batches(self, directory: Union[str, Path],
                            pattern: str = "*.json") -> pd.DataFrame:
        """Load and concatenate multiple JSON batch files."""
        directory = Path(directory)
        files = list(directory.glob(pattern))

        dataframes = []
        for file_path in sorted(files):
            df = self.load(file_path)
            dataframes.append(df)

        if dataframes:
            return pd.concat(dataframes, ignore_index=True)
        return pd.DataFrame()

    @staticmethod
    def save_dataframe_as_json(df: pd.DataFrame, file_path: Union[str, Path],
                              metadata: Optional[Dict] = None):
        """Save DataFrame as JSON."""
        if metadata:
            json_data = {
                "metadata": metadata,
                "data": df.to_dict('records')
            }
        else:
            json_data = df.to_dict('records')

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)