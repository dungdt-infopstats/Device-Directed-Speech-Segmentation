"""
Main Phase 3 implementation: Speech cleaning pipeline.
Combines force alignment (trimming) and audio concatenation.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

from .speech_cleaning import OptimizedSpeechCleaning
from ..phase3_concatenation.audio_concatenator import AudioConcatenator
from ...core.base_data_handler import BasePhase
from ...data.loaders.json_loader import JSONDataLoader


class Phase3SpeechCleaning(BasePhase):
    """Phase 3: Speech cleaning with force alignment and concatenation."""

    def setup(self):
        """Setup Phase 3 components."""
        self.speech_cleaner = OptimizedSpeechCleaning(self.config.get('phase3', {}))
        self.audio_concatenator = AudioConcatenator(self.config.get('phase3', {}))
        self.json_loader = JSONDataLoader()

    def run(self, input_dir: str = None, phase2_data: Optional[pd.DataFrame] = None,
            output_dir: str = None, enable_force_alignment: bool = True) -> Dict[str, Any]:
        """Execute Phase 3 speech cleaning (alignment + concatenation)."""
        if output_dir is None:
            output_dir = self.config.get('data.output_dir') + '/phase3'

        if input_dir is None:
            input_dir = self.config.get('data.output_dir') + '/phase2'

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different outputs
        trimmed_dir = output_dir / 'trimmed_speech'
        concat_dir = output_dir / 'concatenated_speech'

        results = {
            'force_alignment': None,
            'concatenation': None,
            'total_execution_time': 0
        }

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
            print("No metadata found for speech cleaning")
            return results

        metadata_df = pd.DataFrame(metadata_list)

        # Step 1: Force Alignment (Trimming)
        if enable_force_alignment:
            print("Step 1: Running force alignment and trimming...")
            trimmed_dir.mkdir(exist_ok=True)

            alignment_results = self.speech_cleaner.process((
                metadata_df,
                str(input_dir),
                str(trimmed_dir),
                4  # batch_size
            ))
            results['force_alignment'] = alignment_results

            # Use trimmed audio for concatenation
            audio_source_dir = str(trimmed_dir)
            print(f"Force alignment completed: {alignment_results['total_files'] - alignment_results['failed_files']} files processed")
        else:
            # Use original audio from Phase 2
            audio_source_dir = str(input_dir)
            print("Skipping force alignment, using original audio")

        # Step 2: Audio Concatenation
        print("Step 2: Running audio concatenation...")
        concat_dir.mkdir(exist_ok=True)

        concat_results = self.audio_concatenator.process((
            metadata_df,
            audio_source_dir,
            str(concat_dir),
            phase2_data
        ))
        results['concatenation'] = concat_results

        # Save combined results
        if concat_results:
            results_file = output_dir / 'speech_cleaning_results.json'

            # Combine force alignment and concatenation results
            combined_results = {
                'phase': 'speech_cleaning',
                'force_alignment_enabled': enable_force_alignment,
                'force_alignment_results': results['force_alignment'],
                'concatenation_results': len(concat_results),
                'input_dir': str(input_dir),
                'output_dir': str(output_dir),
                'trimmed_dir': str(trimmed_dir) if enable_force_alignment else None,
                'concat_dir': str(concat_dir)
            }

            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(combined_results, f, indent=2, ensure_ascii=False, default=str)

        print(f"Speech cleaning completed: {len(concat_results)} files processed")
        return results

    def cleanup(self):
        """Cleanup Phase 3 resources."""
        if hasattr(self, 'speech_cleaner'):
            self.speech_cleaner.cleanup()