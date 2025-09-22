"""
Speech synthesis using F5-TTS.
Refactored version with abstract TTS handling.
"""

import os
import json
import ast
import soundfile as sf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import time

from .reference_cache import ReferenceCache
from ...core.base_data_handler import BaseProcessor


class SpeechSynthesis(BaseProcessor):
    """Speech synthesis using F5-TTS with reference-based generation."""

    def __init__(self, config: Dict):
        self.config = config
        self.ref_folder_path = config.get('ref_folder_path')
        self.json_folder_path = config.get('json_folder_path')
        self.num_workers = config.get('num_workers', 4)
        self.model = None

    def load_model(self):
        """Load F5-TTS model."""
        try:
            from f5_tts.api import F5TTS
            device = self.config.get('model_config', {}).get('device', 'cuda')
            self.model = F5TTS(device=device)
            return self.model
        except ImportError:
            raise ImportError("f5_tts package not found. Please install F5-TTS.")

    def audio_generate(self, command: str, model: Any, ref_file: str = "", ref_text: str = ""):
        """Generate audio using F5-TTS."""
        wav, sr, spec = model.infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=command,
            seed=None,
        )
        return wav, sr, spec

    def process_single_command(self, row: pd.Series, ref_cache: ReferenceCache,
                             export_path: str, model: Any) -> Dict:
        """Process single command and generate speech."""
        try:
            # Determine target length
            target_len = len(row['text'].split())

            if pd.notna(row['segments']) and row['segments'] not in [None, "", "nan"]:
                seg_list = ast.literal_eval(row['segments'])
                if isinstance(seg_list, list) and len(seg_list) > 0:
                    seg_lens = []
                    for seg in seg_list:
                        try:
                            seg_text = list(seg.values())[0]
                            seg_lens.append(len(seg_text.split()))
                        except Exception:
                            continue
                    if len(seg_lens) > 0:
                        target_len = int(np.mean(seg_lens))

            # Choose appropriate reference
            ref_id, ref_speech_path, ref_text = ref_cache.sample_reference(target_len)

            # Generate full audio
            wav, sr, _ = self.audio_generate(row['text'], model, ref_speech_path, ref_text)

            cur_id = row['id']
            cmd_path = Path(export_path) / f"{row['type']}_{cur_id}"
            cmd_path.mkdir(parents=True, exist_ok=True)

            # Save full audio
            sf.write(cmd_path / f"{row['type']}_{cur_id}_full.wav", wav, sr)

            type_segments = {}
            text_segments = {}

            # Generate individual segments
            if pd.notna(row['segments']) and row['segments'] not in [None, "", "nan"]:
                seg_list = ast.literal_eval(row['segments'])
                for count, seg in enumerate(seg_list):
                    key = list(seg.keys())[0]
                    seg_text = seg[key] + ", "  # Add pause
                    seg_wav, seg_sr, _ = self.audio_generate(seg_text, model, ref_speech_path, ref_text)
                    sf.write(cmd_path / f"{row['type']}_{cur_id}_seg_{count}.wav", seg_wav, seg_sr)

                    # Store segment metadata
                    type_segments[key] = seg.get('type', '')
                    text_segments[key] = seg[key]

            # Save metadata JSON
            json_template = {
                "id": cur_id,
                "type": row['type'],
                "command": row['text'],
                "sampling_rate": sr,
                "num_segments": len(type_segments),
                "type_segments": type_segments,
                "text_segments": text_segments,
                "ref_id": ref_id,
                "ref_file": ref_speech_path,
                "ref_text": ref_text
            }

            with open(cmd_path / f"{row['type']}_{cur_id}.json", 'w', encoding='utf-8') as f:
                json.dump(json_template, f, ensure_ascii=False, indent=2)

            return json_template

        except Exception as e:
            print(f"Failed to process: {row['text']}, error: {e}")
            return None

    def worker_task(self, rows: List[Dict], export_path: str) -> List[Dict]:
        """Worker task for parallel processing."""
        model = self.load_model()
        ref_cache = ReferenceCache(self.ref_folder_path, self.json_folder_path)
        results = []

        for row_dict in rows:
            row = pd.Series(row_dict)
            result = self.process_single_command(row, ref_cache, export_path, model)
            if result:
                results.append(result)

        return results

    def process(self, input_data: Tuple[pd.DataFrame, str]) -> pd.DataFrame:
        """Process input DataFrame and generate speech."""
        text_data, export_path = input_data

        start_time = time.time()
        os.makedirs(export_path, exist_ok=True)

        # Split data into chunks for parallel processing
        chunks = np.array_split(text_data, self.num_workers)

        all_results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(
                    self.worker_task,
                    chunk.to_dict('records'),
                    export_path
                )
                for chunk in chunks
            ]

            for i, future in enumerate(futures):
                results = future.result()
                all_results.extend(results)
                print(f"Worker {i+1} completed ({len(results)} items)")

        print(f"Speech synthesis completed in {time.time() - start_time:.2f}s")
        return pd.DataFrame(all_results) if all_results else pd.DataFrame()

    def get_config(self) -> Dict:
        """Return processor configuration."""
        return self.config