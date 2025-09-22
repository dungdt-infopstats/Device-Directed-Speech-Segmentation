"""
Noise augmentation with text label integration.
Refactored version with abstract noise handling.
"""

import os
import json
import random
import numpy as np
import torch
import torchaudio
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional

from ...core.base_data_handler import BaseProcessor
from ...utils.text_utils import get_active_text_labels, extract_file_id_from_path


class NoiseAugmentor(BaseProcessor):
    """Noise augmentation processor with text label integration."""

    def __init__(self, config: Dict):
        self.config = config
        self.noise_files = []
        self._load_noise_files()

        self.sr = config.get('target_sr', 16000)
        self.snr_range = config.get('snr_range', [-5, 20])
        self.augmentation_ranges = config.get('augmentation_ranges', {
            'prepend_range': [0.5, 2],
            'append_range': [0.5, 2],
            'overlap_range': [0.5, 2]
        })
        self.ratio_range = config.get('ratio_range', True)
        self.min_range_s = config.get('min_range_s', 1)

        # Set random seeds if provided
        seed = config.get('seed')
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def _load_noise_files(self):
        """Load noise files from configured directories."""
        noise_folders = self.config.get('noise_folders', [])
        for folder in noise_folders:
            folder_path = Path(folder)
            if folder_path.exists():
                for f in folder_path.glob("*.wav"):
                    self.noise_files.append(str(f))

        if not self.noise_files:
            print("Warning: No noise files found!")

    def _sec2samples(self, sec: float) -> int:
        """Convert seconds to samples."""
        return int(sec * self.sr)

    def _load_audio(self, path: str) -> torch.Tensor:
        """Load audio file and convert to mono."""
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0, keepdim=True)  # Convert to mono
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return wav.squeeze(0)

    def _scale_noise(self, clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Scale noise to achieve target SNR."""
        Px = clean.pow(2).mean()
        Pn = noise.pow(2).mean() + 1e-12
        k = torch.sqrt(Px / (Pn * (10**(snr_db/10))))
        return noise * k

    def _sample_noise_file(self) -> Tuple[torch.Tensor, str]:
        """Sample random noise file."""
        noise_file = random.choice(self.noise_files)
        return self._load_audio(noise_file), noise_file

    def _sample_noise_segment(self, noise: torch.Tensor, dur_samples: int) -> torch.Tensor:
        """Sample noise segment of specified duration."""
        if noise.shape[-1] <= dur_samples:
            reps = (dur_samples // noise.shape[-1]) + 1
            noise = noise.repeat(reps)[:dur_samples]
            return noise
        else:
            start = random.randint(0, noise.shape[-1] - dur_samples)
            return noise[start:start+dur_samples]

    def _shift_labels(self, labels: List[Dict], shift_s: float) -> List[Dict]:
        """Shift label timestamps by given amount."""
        shifted = []
        for l in labels:
            shifted.append({
                "label": l["label"],
                "start": l["start"] + shift_s,
                "end": l["end"] + shift_s
            })
        return shifted

    def _append_noise_label(self, labels: List[Dict], start_s: float, end_s: float) -> List[Dict]:
        """Add noise label to label list."""
        labels = deepcopy(labels)
        labels.append({"label": "noise", "start": start_s, "end": end_s})
        return labels

    def _merge_overlap_labels(self, labels: List[Dict], noise_start: float, noise_end: float) -> List[Dict]:
        """Merge overlapping noise labels with existing labels."""
        new_labels = deepcopy(labels)
        add_noise = []

        for l in labels:
            if l["label"] == "active":
                if noise_end <= l["start"] or noise_start >= l["end"]:
                    continue  # No overlap
                if noise_start < l["start"]:
                    add_noise.append({"label": "noise", "start": noise_start, "end": l["start"]})
                if noise_end > l["end"]:
                    add_noise.append({"label": "noise", "start": l["end"], "end": noise_end})

        new_labels.extend(add_noise)
        return new_labels

    def _load_labels(self, json_path: str) -> List[Dict]:
        """Load labels from concatenation JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle new format with text_labels and segments
        if isinstance(data, dict) and 'segments' in data:
            return data['segments']
        # Handle old format (list of segments)
        elif isinstance(data, list):
            return data
        else:
            return []

    def _pick_duration(self, base_dur: float, range_vals: List[float]) -> float:
        """Pick duration based on ratio or absolute values."""
        if self.ratio_range:
            min_ratio, max_ratio = range_vals
            dur_s = random.uniform(min_ratio * base_dur, max_ratio * base_dur)
        else:
            dur_s = random.uniform(*range_vals)

        return max(dur_s, self.min_range_s)

    def augment(self, clean_path: str, labels_path: str, mode: str = None,
                custom_snr: float = None, custom_noise: torch.Tensor = None,
                custom_noise_path: str = None, phase2_data: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """Augment audio with noise and return enhanced metadata."""
        labels = self._load_labels(labels_path)
        clean = self._load_audio(clean_path)
        speech_len = clean.shape[-1]
        speech_dur = speech_len / self.sr

        if custom_noise is not None:
            noise = custom_noise
            noise_path = custom_noise_path
        else:
            noise, noise_path = self._sample_noise_file()

        if mode is None:
            mode = random.choice(["overlap", "prepend", "append"])

        snr_db = custom_snr if custom_snr is not None else random.uniform(*self.snr_range)

        if mode == "prepend":
            dur_s = self._pick_duration(speech_dur, self.augmentation_ranges['prepend_range'])
            dur_samples = self._sec2samples(dur_s)
            seg = self._sample_noise_segment(noise, dur_samples)
            out = torch.cat([seg, clean])
            new_labels = self._shift_labels(labels, dur_s)
            new_labels = self._append_noise_label(new_labels, 0, dur_s)

        elif mode == "append":
            dur_s = self._pick_duration(speech_dur, self.augmentation_ranges['append_range'])
            dur_samples = self._sec2samples(dur_s)
            seg = self._sample_noise_segment(noise, dur_samples)
            out = torch.cat([clean, seg])
            new_labels = deepcopy(labels)
            new_labels = self._append_noise_label(new_labels, speech_dur, speech_dur + dur_s)

        elif mode == "overlap":
            dur_s = self._pick_duration(speech_dur, self.augmentation_ranges['overlap_range'])
            dur_samples = min(self._sec2samples(dur_s), speech_len)
            offset = random.randint(0, speech_len - dur_samples)
            offset_s = offset / self.sr
            seg = self._sample_noise_segment(noise, dur_samples)

            noise_aligned = torch.zeros_like(clean)
            noise_aligned[offset:offset+dur_samples] = seg
            noise_scaled = self._scale_noise(clean, noise_aligned, snr_db)
            out = clean + noise_scaled

            noise_start = offset_s
            noise_end = offset_s + dur_s
            new_labels = self._merge_overlap_labels(labels, noise_start, noise_end)

        # Get text labels from Phase 2 data or labels file
        text_labels = ""
        if phase2_data:
            file_id = extract_file_id_from_path(Path(clean_path).parent.name)
            text_labels = get_active_text_labels(phase2_data, file_id)
        else:
            # Try to extract from labels file if it has text_labels field
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    label_data = json.load(f)
                if isinstance(label_data, dict):
                    text_labels = label_data.get('text_labels', '')
            except:
                pass

        metadata = {
            "clean_path": clean_path,
            "noise_path": noise_path,
            "mode": mode,
            "snr_db": snr_db,
            "text_labels": text_labels,
            "labels": new_labels,
        }

        return out, metadata

    def process(self, input_data) -> List[Dict]:
        """Process noise augmentation for input data."""
        input_dir, output_dir, phase2_data, custom_mode = input_data
        results = []

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Process each audio file
        for file_folder in input_path.iterdir():
            if not file_folder.is_dir():
                continue

            audio_files = list(file_folder.glob("*_concat.wav"))
            if not audio_files:
                continue

            audio_file = audio_files[0]
            file_name = file_folder.name
            label_file = file_folder / f"{file_name}.json"

            if not label_file.exists():
                print(f"Label file not found: {label_file}")
                continue

            try:
                # Basic augmentation
                out, meta = self.augment(
                    clean_path=str(audio_file),
                    labels_path=str(label_file),
                    phase2_data=phase2_data
                )

                # Save basic augmented file
                out_folder = output_path / file_name
                out_folder.mkdir(exist_ok=True)

                out_audio_path = out_folder / f"{file_name}_aug.wav"
                import soundfile as sf
                sf.write(str(out_audio_path), out.numpy(), self.sr)

                out_json_path = out_folder / f"{file_name}_aug.json"
                with open(out_json_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

                results.append({
                    "file_name": file_name,
                    "audio_path": str(out_audio_path),
                    "metadata_path": str(out_json_path),
                    "text_labels": meta["text_labels"]
                })

                # Custom mode with multiple SNR ranges
                if custom_mode:
                    snr_ranges = [(-5, 0), (0, 5), (5, 10), (10, 15), (15, 20)]
                    for i in range(5):
                        custom_noise, custom_noise_path = self._sample_noise_file()
                        for idx, snr_range in enumerate(snr_ranges):
                            custom_snr_db = random.uniform(*snr_range)
                            out_custom, meta_custom = self.augment(
                                clean_path=str(audio_file),
                                labels_path=str(label_file),
                                custom_snr=custom_snr_db,
                                custom_noise=custom_noise,
                                custom_noise_path=custom_noise_path,
                                phase2_data=phase2_data
                            )

                            out_audio_custom = out_folder / f"{file_name}_aug_{i}-{idx}.wav"
                            sf.write(str(out_audio_custom), out_custom.numpy(), self.sr)

                            out_json_custom = out_folder / f"{file_name}_aug_{i}-{idx}.json"
                            with open(out_json_custom, "w", encoding="utf-8") as f:
                                json.dump(meta_custom, f, indent=2, ensure_ascii=False)

                            results.append({
                                "file_name": f"{file_name}_{i}-{idx}",
                                "audio_path": str(out_audio_custom),
                                "metadata_path": str(out_json_custom),
                                "text_labels": meta_custom["text_labels"]
                            })

            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                continue

        return results

    def get_config(self) -> Dict:
        """Return processor configuration."""
        return self.config