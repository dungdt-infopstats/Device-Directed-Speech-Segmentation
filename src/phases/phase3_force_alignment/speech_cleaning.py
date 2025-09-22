"""
Speech cleaning with force alignment using Whisper.
Refactored version with abstract handling and optimizations.
"""

import os
import json
import time
import logging
import pandas as pd
import numpy as np
import torch
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import re

from ...core.base_data_handler import BaseProcessor


class OptimizedSpeechCleaning(BaseProcessor):
    """
    Optimized speech cleaning processor with force alignment.

    Uses faster-whisper for improved performance and includes
    memory management and parallel processing.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.cant_clean_list = config.get('cant_clean_list', [])
        self.max_workers = config.get('max_workers', 4)
        self.use_gpu = config.get('use_gpu', True)
        self.model_size = config.get('model_size', 'medium')
        self.padding = config.get('padding', 0.3)

        self.model = None
        self.device = None
        self._setup_device()
        self._setup_logging()

    def _setup_device(self):
        """Setup device and compute type."""
        if self.use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            logging.info("Using CPU")

    def _setup_logging(self):
        """Setup logging for the speech cleaning process."""
        self.logger = logging.getLogger('speech_cleaning')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _load_model(self):
        """Load faster-whisper model with optimizations."""
        try:
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                cpu_threads=os.cpu_count() if self.device == "cpu" else 4
            )
            self.logger.info(f"Loaded {self.model_size} Whisper model on {self.device}")
            return self.model
        except ImportError:
            raise ImportError("faster-whisper package not found. Please install: pip install faster-whisper")

    def normalize_text_cached(self, text: str) -> str:
        """Normalize text for alignment."""
        return re.sub(r'[^\w\s]', '', text.lower()).strip()

    def word_similarity_fast(self, w1: str, w2: str) -> float:
        """Fast word similarity calculation."""
        return SequenceMatcher(None, w1, w2).ratio()

    def smith_waterman_fuzzy_optimized(self, ref_words: List[str], hypo_words: List[str],
                                     match_score=2, fuzzy_score=1, mismatch=-1, gap=-2):
        """Optimized Smith-Waterman algorithm with numpy."""
        m, n = len(ref_words), len(hypo_words)

        if m == 0 or n == 0:
            raise ValueError("Empty word lists provided")

        # Use numpy for better performance
        score = np.zeros((m+1, n+1), dtype=np.float32)
        max_score = 0
        max_pos = None

        # Pre-compute similarities
        similarities = np.zeros((m, n), dtype=np.float32)
        for i in range(m):
            for j in range(n):
                similarities[i, j] = self.word_similarity_fast(ref_words[i], hypo_words[j])

        for i in range(1, m+1):
            for j in range(1, n+1):
                sim = similarities[i-1, j-1]

                if sim == 1:
                    s = match_score
                elif sim >= 0.8:
                    s = fuzzy_score
                else:
                    s = mismatch

                diag = score[i-1, j-1] + s
                delete = score[i-1, j] + gap
                insert = score[i, j-1] + gap
                score[i, j] = max(0, diag, delete, insert)

                if score[i, j] > max_score:
                    max_score = score[i, j]
                    max_pos = (i, j)

        if max_pos is None:
            raise ValueError("No alignment found")

        # Traceback
        i, j = max_pos
        end_j = j - 1

        while i > 0 and j > 0 and score[i, j] > 0:
            sim = similarities[i-1, j-1]
            if sim >= 0.8:
                i -= 1
                j -= 1
            elif score[i-1, j] + gap == score[i, j]:
                i -= 1
            else:
                j -= 1

        start_j = j
        return start_j, end_j

    def get_start_end_from_alignment(self, ref_text: str, whisper_words: List[Dict]) -> Tuple[float, float]:
        """Get start and end times from alignment."""
        ref_words = self.normalize_text_cached(ref_text).split()
        hypo_words = [self.normalize_text_cached(w['word']) for w in whisper_words]

        if not ref_words or not hypo_words:
            raise ValueError("Empty reference or hypothesis words")

        start_idx, end_idx = self.smith_waterman_fuzzy_optimized(ref_words, hypo_words)

        if start_idx >= len(whisper_words) or end_idx >= len(whisper_words):
            raise ValueError("Alignment indices out of bounds")

        start_time = whisper_words[start_idx]['start']
        end_time = whisper_words[end_idx]['end']

        return start_time, end_time

    def get_clean_range(self, file_name: str, file_type: str, speech_folder: str) -> Tuple[Optional[float], Optional[float]]:
        """Get clean audio range using Whisper transcription."""
        file_path = os.path.join(speech_folder, file_name, f"{file_name}_{file_type}.wav")

        if not os.path.exists(file_path):
            self.logger.warning(f"File not found: {file_path}")
            return None, None

        try:
            # Use faster-whisper for transcription
            segments, info = self.model.transcribe(
                file_path,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                beam_size=1,
                language="vi" if "vietnamese" in file_path.lower() else None
            )

            # Extract words with timestamps
            whisper_words = []
            for segment in segments:
                for word in segment.words:
                    whisper_words.append({
                        'word': word.word,
                        'start': word.start,
                        'end': word.end
                    })

            if not whisper_words:
                self.logger.warning(f"{file_name} {file_type} - No words transcribed")
                self.cant_clean_list.append((file_name, file_type, "no words transcribed"))
                return None, None

        except Exception as e:
            self.logger.error(f"{file_name} {file_type} - Transcription error: {e}")
            self.cant_clean_list.append((file_name, file_type, "cant transcribe"))
            return None, None

        # Load reference text
        try:
            meta_path = os.path.join(speech_folder, file_name, f"{file_name}.json")
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if file_type == "full":
                ref_text = data['command']
            else:
                num_seg = file_type.split("_")[1]
                ref_text = data['text_segments'][num_seg]

        except Exception as e:
            self.logger.error(f"Error loading reference text for {file_name}: {e}")
            return None, None

        try:
            start, end = self.get_start_end_from_alignment(ref_text, whisper_words)

            if self.padding:
                start = max(0, start - self.padding)
                end = end + self.padding

            return start, end

        except Exception as e:
            self.logger.error(f"{file_name} {file_type} - Alignment error: {e}")
            self.cant_clean_list.append((file_name, file_type, "cant get start end"))
            return None, None

    def trim_audio(self, file_name: str, file_type: str, start: float, end: float, speech_folder: str):
        """Trim audio using pydub."""
        try:
            from pydub import AudioSegment

            file_path = os.path.join(speech_folder, file_name, f"{file_name}_{file_type}.wav")

            if not os.path.exists(file_path):
                self.logger.warning(f"Audio file not found: {file_path}")
                return AudioSegment.empty()

            audio = AudioSegment.from_wav(file_path)

            if start is not None and end is not None:
                start_ms = int(start * 1000)
                end_ms = int(end * 1000)
                trimmed_audio = audio[start_ms:end_ms]
                return trimmed_audio
            else:
                return audio

        except Exception as e:
            self.logger.error(f"Error trimming audio {file_name}_{file_type}: {e}")
            return AudioSegment.empty()

    def _process_single_file(self, export_dir: str, row: pd.Series, speech_folder: str):
        """Process a single file for speech cleaning."""
        file_name = f"{row['type']}_{row['id']}"

        # Process full file
        self._process_single_segment(export_dir, file_name, "full", speech_folder, row)

        # Process segments if applicable
        if row['type'] in ['single_mix', 'chain_mix']:
            for i in range(row['num_segments']):
                self._process_single_segment(export_dir, file_name, f"seg_{i}", speech_folder, row)

    def _process_single_segment(self, export_dir: str, file_name: str, file_type: str, speech_folder: str, row: pd.Series):
        """Process a single audio segment."""
        try:
            start, end = self.get_clean_range(file_name, file_type, speech_folder)

            if start is not None and end is not None:
                trimmed_audio = self.trim_audio(file_name, file_type, start, end, speech_folder)

                # Create output directory
                output_dir = Path(export_dir) / file_name
                output_dir.mkdir(exist_ok=True)

                out_path = output_dir / f"{file_name}_{file_type}_trimmed.wav"
                trimmed_audio.export(str(out_path), format="wav")

                self.logger.debug(f"Trimmed {file_name}_{file_type}")

        except Exception as e:
            self.logger.error(f"Error processing {file_name}_{file_type}: {e}")
            self.cant_clean_list.append((file_name, file_type, str(e)))

    def clean_pipeline(self, export_dir: str, data: pd.DataFrame, speech_folder: str, batch_size: int = 4):
        """Main cleaning pipeline with batch processing."""
        os.makedirs(export_dir, exist_ok=True)

        # Load model if not already loaded
        if self.model is None:
            self._load_model()

        # Split data into batches
        batches = [data.iloc[i:i+batch_size] for i in range(0, len(data), batch_size)]

        total_processed = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(batches):
            self.logger.info(f"Processing batch {batch_idx + 1}/{len(batches)}")

            # Process batch with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []

                for idx, row in batch.iterrows():
                    future = executor.submit(
                        self._process_single_file,
                        export_dir, row, speech_folder
                    )
                    futures.append(future)

                # Collect results
                for future in futures:
                    try:
                        future.result(timeout=300)  # 5 minutes timeout
                        total_processed += 1
                    except Exception as e:
                        self.logger.error(f"Error processing file: {e}")

            # Memory cleanup after each batch
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        end_time = time.time()
        self.logger.info(f"Processed {total_processed} files in {end_time - start_time:.2f} seconds")

        if total_processed > 0:
            self.logger.info(f"Average: {(end_time - start_time) / total_processed:.2f} seconds per file")

    def process(self, input_data) -> Dict[str, Any]:
        """Process speech cleaning."""
        data, speech_folder, export_dir, batch_size = input_data

        start_time = time.time()
        self.clean_pipeline(export_dir, data, speech_folder, batch_size)

        return {
            'total_files': len(data),
            'failed_files': len(self.cant_clean_list),
            'processing_time': time.time() - start_time,
            'cant_clean_list': self.cant_clean_list
        }

    def get_config(self) -> Dict[str, Any]:
        """Return processor configuration."""
        return {
            **self.config,
            'device': self.device,
            'model_size': self.model_size,
            'max_workers': self.max_workers,
            'failed_files': len(self.cant_clean_list)
        }

    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'model') and self.model:
            del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()