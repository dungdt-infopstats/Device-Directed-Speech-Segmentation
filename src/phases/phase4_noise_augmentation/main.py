"""
Main Phase 4 implementation: Noise augmentation pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

from .noise_augmentor import NoiseAugmentor
from ...core.base_data_handler import BasePhase
from ...data.loaders.json_loader import JSONDataLoader


class Phase4NoiseAugmentation(BasePhase):
    """Phase 4: Noise augmentation with text label integration."""

    def setup(self):
        """Setup Phase 4 components."""
        self.noise_augmentor = NoiseAugmentor(self.config.get('phase4', {}))
        self.json_loader = JSONDataLoader()

    def run(self, input_dir: str = None, output_dir: str = None,
            phase2_data: Optional[pd.DataFrame] = None,
            custom_mode: bool = False) -> List[Dict]:
        """Execute Phase 4 noise augmentation."""
        if output_dir is None:
            output_dir = self.config.get('data.output_dir') + '/phase4'

        if input_dir is None:
            input_dir = self.config.get('data.output_dir') + '/phase3'

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load Phase 2 data if not provided (for text labels)
        if phase2_data is None:
            phase2_dir = Path(self.config.get('data.output_dir') + '/phase2')
            phase2_results_file = phase2_dir / 'speech_synthesis_results.json'
            if phase2_results_file.exists():
                phase2_data = self.json_loader.load(phase2_results_file)
            else:
                print("Warning: No Phase 2 data found for text label extraction")
                phase2_data = pd.DataFrame()

        # Process noise augmentation
        results = self.noise_augmentor.process((
            str(input_dir),
            str(output_dir),
            phase2_data,
            custom_mode
        ))

        # Save augmentation results
        if results:
            results_file = output_dir / 'augmentation_results.json'
            self.json_loader.save_dataframe_as_json(
                pd.DataFrame(results),
                results_file,
                metadata={
                    "phase": "noise_augmentation",
                    "total_records": len(results),
                    "input_dir": str(input_dir),
                    "output_dir": str(output_dir),
                    "custom_mode": custom_mode,
                    "config": self.noise_augmentor.get_config()
                }
            )

        return results

    def cleanup(self):
        """Cleanup Phase 4 resources."""
        pass