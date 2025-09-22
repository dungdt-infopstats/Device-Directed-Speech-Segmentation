"""
Content data loader for apps, movies, songs, and TV shows.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union
from .json_loader import JSONDataLoader
from ...core.base_data_handler import BaseDataLoader


class ContentLoader(BaseDataLoader):
    """Loader for content data (apps, movies, songs, TV shows)."""

    def __init__(self, content_dir: Union[str, Path]):
        self.content_dir = Path(content_dir)
        self.json_loader = JSONDataLoader()

    def load(self, source: Union[str, Path] = None) -> pd.DataFrame:
        """Load all content files and return combined DataFrame."""
        if source:
            return self.json_loader.load(source)

        content_dict = self.load_all_content()

        # Combine all content into single DataFrame
        all_content = []
        for content_type, items in content_dict.items():
            for item in items:
                all_content.append({
                    'content': item['content'],
                    'content_type': content_type
                })

        return pd.DataFrame(all_content)

    def load_all_content(self) -> Dict[str, List[Dict]]:
        """Load all content files and return as dictionary."""
        content_dict = {}

        for content_file in ['app.json', 'movie.json', 'song.json', 'tv.json']:
            file_path = self.content_dir / content_file
            if file_path.exists():
                content_type = content_file.replace('.json', '')
                df = self.json_loader.load(file_path)
                content_dict[content_type] = df.to_dict('records')

        return content_dict

    def validate(self, data: pd.DataFrame) -> bool:
        """Validate content data structure."""
        required_columns = ['content']
        return all(col in data.columns for col in required_columns)

    def get_content_by_type(self, content_type: str) -> List[str]:
        """Get content list for specific type."""
        file_path = self.content_dir / f"{content_type}.json"
        if not file_path.exists():
            return []

        df = self.json_loader.load(file_path)
        return df['content'].tolist()