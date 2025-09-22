"""
Main Phase 2 implementation: Speech synthesis pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from .speech_synthesis import SpeechSynthesis
from ...core.base_data_handler import BasePhase
from ...data.loaders.json_loader import JSONDataLoader


class Phase2SpeechSynthesis(BasePhase):
    """Phase 2: Speech synthesis using F5-TTS."""

    def setup(self):
        """Setup Phase 2 components."""
        self.speech_synthesis = SpeechSynthesis(self.config.get('phase2'))
        self.json_loader = JSONDataLoader()

    def run(self, input_data: pd.DataFrame = None, input_dir: str = None,
            output_dir: str = None) -> pd.DataFrame:
        """Execute Phase 2 speech synthesis."""
        if output_dir is None:
            output_dir = self.config.get('data.output_dir') + '/phase2'

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load input data if not provided
        if input_data is None:
            if input_dir is None:
                input_dir = self.config.get('data.output_dir') + '/phase1'
            input_data = self.json_loader.load_multiple_batches(input_dir)

        if input_data.empty:
            print("No input data found for Phase 2")
            return pd.DataFrame()

        # Process speech synthesis
        results_df = self.speech_synthesis.process((input_data, str(output_dir)))

        if not results_df.empty:
            # Save results
            output_file = output_dir / 'speech_synthesis_results.json'
            self.json_loader.save_dataframe_as_json(
                results_df,
                output_file,
                metadata={
                    "phase": "speech_synthesis",
                    "total_records": len(results_df),
                    "config": self.speech_synthesis.get_config()
                }
            )

        return results_df

    def cleanup(self):
        """Cleanup Phase 2 resources."""
        pass