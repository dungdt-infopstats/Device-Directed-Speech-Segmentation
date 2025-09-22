"""
Audio concatenation with text label integration.
Refactored version with abstract audio handling.
"""

import os
import json
import ast
import pandas as pd
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Set, Union, Optional
from pydub import AudioSegment

from ...core.base_data_handler import BaseProcessor
from ...utils.text_utils import get_active_text_labels, extract_file_id_from_path


class AudioConcatenator(BaseProcessor):
    """Audio concatenation processor with text label integration."""

    def __init__(self, config: Dict):
        self.config = config
        self.bad_list = set(config.get('bad_list', []))
        self.post_fix = config.get('post_fix', '')

    def _parse_segments(self, type_segments: Union[str, Dict]) -> Dict:
        """Parse type segments from string or dict format."""
        if isinstance(type_segments, str):
            return ast.literal_eval(type_segments)
        if isinstance(type_segments, dict):
            return type_segments
        raise ValueError("Unknown type_segments format")

    def _concat_file(self, row: pd.Series, input_folder: str, output_folder: str,
                    phase2_data: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        """Concatenate audio segments for mix types."""
        file_name = f"{row['type']}_{row['id']}"
        type_segments = self._parse_segments(row['type_segments'])

        combined = AudioSegment.silent(duration=0)
        annotations = []
        current_time_ms = 0
        num_segments = row['num_segments']

        for i in range(num_segments):
            seg_path = os.path.join(input_folder, file_name, f"{file_name}_seg_{i}{self.post_fix}.wav")
            if not os.path.exists(seg_path):
                print(f"Missing: {seg_path}")
                continue

            segment = AudioSegment.from_wav(seg_path)
            duration_ms = len(segment)

            label = type_segments.get(str(i), "unknown")
            annotations.append({
                "label": label,
                "start": current_time_ms / 1000.0,
                "end": (current_time_ms + duration_ms) / 1000.0
            })

            combined += segment
            current_time_ms += duration_ms

        return self._save_outputs(file_name, combined, annotations, output_folder, phase2_data)

    def _get_non_mix(self, row: pd.Series, input_folder: str, output_folder: str,
                    phase2_data: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        """Process non-mix audio files (single_active, chain_active, non_active)."""
        file_name = f"{row['type']}_{row['id']}"
        file_path = os.path.join(input_folder, file_name, f"{file_name}{self.post_fix}.wav")

        if not os.path.exists(file_path):
            file_name_full = f"{row['type']}_{row['id']}_full"
            file_path = os.path.join(input_folder, file_name, f"{file_name_full}{self.post_fix}.wav")

        if not os.path.exists(file_path):
            print(f"Missing: {file_path}")
            return None

        segment = AudioSegment.from_wav(file_path)
        duration_ms = len(segment)

        label = "non_active" if row["type"] == "non_active" else "active"
        annotations = [{
            "label": label,
            "start": 0,
            "end": duration_ms / 1000.0
        }]

        return self._save_outputs(file_name, segment, annotations, output_folder, phase2_data)

    def _save_outputs(self, file_name: str, audio: AudioSegment, annotations: List[Dict],
                     output_folder: str, phase2_data: Optional[pd.DataFrame] = None) -> Dict:
        """Save audio and enhanced annotations with text labels."""
        file_output_dir = os.path.join(output_folder, file_name)
        os.makedirs(file_output_dir, exist_ok=True)

        # Save audio
        output_wav = os.path.join(file_output_dir, f"{file_name}_concat.wav")
        audio.export(output_wav, format="wav")

        # Get active text labels
        text_labels = ""
        if phase2_data is not None:
            file_id = extract_file_id_from_path(file_name)
            text_labels = get_active_text_labels(phase2_data, file_id)

        # Create enhanced JSON structure with text labels
        output_data = {
            "text_labels": text_labels,
            "segments": annotations
        }

        # Save annotations
        output_json = os.path.join(file_output_dir, f"{file_name}.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Saved: {output_wav}, {output_json}")
        return {"wav": output_wav, "json": output_json, "text_labels": text_labels}

    def process(self, input_data) -> List[Dict]:
        """Process audio concatenation for all files."""
        metadata, input_folder, output_folder, phase2_data = input_data
        results = []

        for _, row in metadata.iterrows():
            file_name = f"{row['type']}_{row['id']}"
            file_name_full = f"{row['type']}_{row['id']}_full"

            # Skip if in bad list
            if file_name in self.bad_list or file_name_full in self.bad_list:
                print(f"Skip {file_name}")
                continue

            # Process based on segment count
            if row.get("num_segments", 0) != 0:
                result = self._concat_file(row, input_folder, output_folder, phase2_data)
            else:
                result = self._get_non_mix(row, input_folder, output_folder, phase2_data)

            if result:
                results.append(result)

        return results

    def get_config(self) -> Dict:
        """Return processor configuration."""
        return self.config