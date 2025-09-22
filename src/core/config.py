"""
Configuration management for the synthesis pipeline.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union


class Config:
    """Configuration manager for the synthesis pipeline."""

    def __init__(self, config_path: Union[str, Path] = None):
        self.config_path = Path(config_path) if config_path else None
        self._config = self._load_default_config()

        if self.config_path and self.config_path.exists():
            self.load_config(self.config_path)

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "data": {
                "content_dir": "data/content",
                "commands_dir": "data/commands",
                "output_dir": "data/processed"
            },
            "phase1": {
                "prompt_gen_param": {
                    "num_samples_command": 5,
                    "num_samples_content": 5,
                    "chain_length": 3,
                    "generated_nums": 4
                },
                "text_gen_param": {
                    "model": "gpt-4o-mini",
                    "temperature": 1.0
                },
                "batch_config": {
                    "loop": 5,
                    "file_num": 50,
                    "start": 100
                }
            },
            "phase2": {
                "ref_folder_path": "data/external/vctk/filtered_audio",
                "json_folder_path": "data/external/vctk/filtered_json",
                "num_workers": 4,
                "model_config": {
                    "device": "cuda"
                }
            },
            "phase3": {
                "padding_ms": 1000,
                "max_duration": 12.0,
                "min_duration": 1.0
            },
            "phase4": {
                "noise_folders": [
                    "data/external/musan/speech/us-gov",
                    "data/external/musan/speech/librivox"
                ],
                "target_sr": 16000,
                "snr_range": [-5, 20],
                "augmentation_ranges": {
                    "prepend_range": [0.5, 2],
                    "append_range": [0.5, 2],
                    "overlap_range": [0.5, 2]
                },
                "ratio_range": True,
                "min_range_s": 1
            }
        }

    def load_config(self, config_path: Union[str, Path]):
        """Load configuration from file."""
        config_path = Path(config_path)

        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        self._merge_config(user_config)

    def _merge_config(self, user_config: Dict[str, Any]):
        """Merge user configuration with default configuration."""
        def merge_dict(default: Dict, user: Dict) -> Dict:
            result = default.copy()
            for key, value in user.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result

        self._config = merge_dict(self._config, user_config)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key path (e.g., 'phase1.model')."""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value by key path."""
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save_config(self, config_path: Union[str, Path]):
        """Save current configuration to file."""
        config_path = Path(config_path)

        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        return self._config.copy()