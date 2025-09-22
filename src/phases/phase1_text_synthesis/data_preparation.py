"""
Data preparation for Phase 1 text synthesis.
Refactored version with abstract data handling.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Union
from ...data.loaders.content_loader import ContentLoader
from ...data.loaders.command_loader import CommandLoader
from ...core.base_data_handler import BaseProcessor


class DataPreparation(BaseProcessor):
    """Abstract data preparation for text synthesis."""

    def __init__(self, content_loader: ContentLoader, command_loader: CommandLoader):
        self.content_loader = content_loader
        self.command_loader = command_loader

    def process(self, input_data=None) -> Tuple[Dict[str, List[str]], List[str]]:
        """Process and return content dictionary and commands list."""
        content_dict = self.content_loader.load_all_content()
        commands = self.command_loader.get_commands_list()

        return content_dict, commands

    def get_config(self) -> Dict[str, any]:
        """Return processor configuration."""
        return {
            "content_dir": str(self.content_loader.content_dir),
            "commands_dir": str(self.command_loader.commands_dir)
        }