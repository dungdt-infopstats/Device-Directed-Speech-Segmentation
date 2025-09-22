"""
Reference cache for managing VCTK reference data.
Refactored version with abstract data handling.
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from ...core.base_data_handler import BaseProcessor


class ReferenceCache(BaseProcessor):
    """Cache and manage reference audio and text data."""

    def __init__(self, audio_folder_path: str, json_folder_path: str):
        self.audio_folder_path = Path(audio_folder_path)
        self.json_folder_path = Path(json_folder_path)
        self._audio_files = None
        self._json_files = None
        self._ref_texts = {}

    @property
    def audio_files(self) -> List[Path]:
        """Get list of audio files."""
        if self._audio_files is None:
            self._audio_files = [f for f in self.audio_folder_path.iterdir() if f.is_file()]
        return self._audio_files

    @property
    def json_files(self) -> List[Path]:
        """Get list of JSON files."""
        if self._json_files is None:
            self._json_files = [f for f in self.json_folder_path.iterdir() if f.suffix == '.json']
        return self._json_files

    def get_ref_text(self, json_path: Path) -> str:
        """Get reference text from JSON file with caching."""
        json_str = str(json_path)
        if json_str not in self._ref_texts:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._ref_texts[json_str] = data.get('text', '')
        return self._ref_texts[json_str]

    def sample_reference(self, target_len: int) -> Tuple[str, str, str]:
        """
        Sample reference based on target length.

        Args:
            target_len: Target text length in words

        Returns:
            Tuple of (ref_id, audio_file_path, ref_text)
        """
        # Sample 30 random JSON files
        sampled_jsons = random.sample(self.json_files, min(30, len(self.json_files)))

        # Calculate distances from target length
        candidates = []
        for json_file in sampled_jsons:
            ref_text = self.get_ref_text(json_file)
            ref_len = len(ref_text.split())
            diff = abs(ref_len - target_len)
            candidates.append((diff, json_file, ref_text))

        # Sort by distance and take top 5
        candidates.sort(key=lambda x: x[0])
        top_candidates = candidates[:5]

        # Random choice from top 5
        _, json_file, ref_text = random.choice(top_candidates)
        ref_id = json_file.stem
        audio_file = self.audio_folder_path / f"{ref_id}.wav"

        return ref_id, str(audio_file), ref_text

    def process(self, input_data) -> Tuple[str, str, str]:
        """Process input and return reference data."""
        if isinstance(input_data, int):
            return self.sample_reference(input_data)
        else:
            raise ValueError("Input must be target length (int)")

    def get_config(self) -> Dict:
        """Return processor configuration."""
        return {
            "audio_folder_path": str(self.audio_folder_path),
            "json_folder_path": str(self.json_folder_path),
            "total_audio_files": len(self.audio_files),
            "total_json_files": len(self.json_files)
        }