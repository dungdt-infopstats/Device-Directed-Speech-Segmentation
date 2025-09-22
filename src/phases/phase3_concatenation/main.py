"""
Main Phase 3 implementation: Audio concatenation pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from .audio_concatenator import AudioConcatenator
from ...core.base_data_handler import BasePhase
from ...data.loaders.json_loader import JSONDataLoader


class Phase3Concatenation(BasePhase):
    """Phase 3: Audio concatenation with text label integration."""

    def setup(self):
        """Setup Phase 3 components."""
        self.audio_concatenator = AudioConcatenator(self.config.get('phase3', {}))
        self.json_loader = JSONDataLoader()

    def run(self, input_dir: str = None, phase2_data: Optional[pd.DataFrame] = None,
            output_dir: str = None) -> List[Dict]:
        """Execute Phase 3 audio concatenation."""
        if output_dir is None:
            output_dir = self.config.get('data.output_dir') + '/phase3'

        if input_dir is None:
            input_dir = self.config.get('data.output_dir') + '/phase2'

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load Phase 2 data if not provided
        if phase2_data is None:
            phase2_results_file = Path(input_dir) / 'speech_synthesis_results.json'
            if phase2_results_file.exists():
                phase2_data = self.json_loader.load(phase2_results_file)
            else:
                print("Warning: No Phase 2 data found for text label extraction")
                phase2_data = pd.DataFrame()

        # Load metadata from Phase 2 directory structure
        metadata_list = []
        input_path = Path(input_dir)

        # Look for individual JSON files in subdirectories
        for subdir in input_path.iterdir():
            if subdir.is_dir():
                for json_file in subdir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            metadata_list.append(data)
                    except Exception as e:
                        print(f"Error reading {json_file}: {e}")

        if not metadata_list:
            print("No metadata found for concatenation")
            return []

        metadata_df = pd.DataFrame(metadata_list)

        # Process concatenation
        results = self.audio_concatenator.process((
            metadata_df,
            str(input_dir),
            str(output_dir),
            phase2_data
        ))

        # Save concatenation results
        if results:
            results_file = output_dir / 'concatenation_results.json'
            self.json_loader.save_dataframe_as_json(
                pd.DataFrame(results),
                results_file,
                metadata={
                    "phase": "audio_concatenation",
                    "total_records": len(results),
                    "input_dir": str(input_dir),
                    "output_dir": str(output_dir)
                }
            )

        return results

    def cleanup(self):
        """Cleanup Phase 3 resources."""
        pass