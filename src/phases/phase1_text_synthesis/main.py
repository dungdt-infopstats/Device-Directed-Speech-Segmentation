"""
Main Phase 1 implementation: Text synthesis pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from tqdm import tqdm

from .data_preparation import DataPreparation
from .prompt_generator import PromptGenerator
from .text_generator import TextGenerator
from ...core.base_data_handler import BasePhase
from ...data.loaders.content_loader import ContentLoader
from ...data.loaders.command_loader import CommandLoader
from ...data.loaders.json_loader import JSONDataLoader


class Phase1TextSynthesis(BasePhase):
    """Phase 1: Text synthesis and generation."""

    def setup(self):
        """Setup Phase 1 components."""
        self.content_loader = ContentLoader(self.config.get('data.content_dir'))
        self.command_loader = CommandLoader(self.config.get('data.commands_dir'))
        self.data_preparation = DataPreparation(self.content_loader, self.command_loader)
        self.prompt_generator = PromptGenerator(self.config.get('phase1.prompt_gen_param'))
        self.text_generator = TextGenerator(self.config.get('phase1.text_gen_param'))
        self.json_loader = JSONDataLoader()

    def run(self, output_dir: str = None) -> pd.DataFrame:
        """Execute Phase 1 text synthesis."""
        if output_dir is None:
            output_dir = self.config.get('data.output_dir') + '/phase1'

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get batch configuration
        batch_config = self.config.get('phase1.batch_config')
        loop = batch_config.get('loop', 5)
        file_num = batch_config.get('file_num', 50)
        start = batch_config.get('start', 100)

        # Prepare data
        content_dict, commands = self.data_preparation.process()

        all_results = []

        for i in tqdm(range(start, start + file_num), desc="Generating text batches"):
            try:
                batch_results = []

                for _ in range(loop):
                    # Generate prompts
                    prompts = self.prompt_generator.process((commands, content_dict))

                    # Generate text
                    df_batch = self.text_generator.process(prompts)
                    batch_results.append(df_batch)

                # Combine batch results
                if batch_results:
                    batch_df = pd.concat(batch_results, ignore_index=True)

                    # Save individual batch
                    batch_file = output_dir / f'batch_{i:03d}.json'
                    self.json_loader.save_dataframe_as_json(
                        batch_df,
                        batch_file,
                        metadata={
                            "batch_id": f"batch_{i:03d}",
                            "phase": "text_synthesis",
                            "total_records": len(batch_df)
                        }
                    )

                    all_results.append(batch_df)

            except Exception as e:
                print(f'Error at batch {i}: {e}')
                continue

        # Combine all results
        if all_results:
            full_df = pd.concat(all_results, ignore_index=True)

            # Save full dataset
            full_file = output_dir / 'full_dataset.json'
            self.json_loader.save_dataframe_as_json(
                full_df,
                full_file,
                metadata={
                    "phase": "text_synthesis",
                    "total_records": len(full_df),
                    "total_batches": len(all_results)
                }
            )

            return full_df

        return pd.DataFrame()

    def cleanup(self):
        """Cleanup Phase 1 resources."""
        pass