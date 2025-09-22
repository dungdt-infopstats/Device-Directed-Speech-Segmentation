"""
Text generator for Phase 1 text synthesis.
Refactored version with abstract LLM handling.
"""

import os
import json
import uuid
import pandas as pd
from typing import Dict, List, Optional
from openai import OpenAI
from ...core.base_data_handler import BaseProcessor


class TextGenerator(BaseProcessor):
    """Generate text using LLM APIs."""

    def __init__(self, config: Dict):
        self.config = config
        self.model = config.get('model', 'gpt-4o-mini')
        self.temperature = config.get('temperature', 1.0)
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            api_key = self._get_api_key()
            return OpenAI(api_key=api_key)
        except Exception as e:
            print(f'Failed to initialize model: {e}')
            raise e

    def _get_api_key(self) -> str:
        """Get API key from environment."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return api_key

    def generate(self, prompt: str) -> str:
        """Generate text using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def create_dataframe(self, parsed_data: Dict) -> pd.DataFrame:
        """Create DataFrame from parsed generation results."""
        rows = []

        for main_type, items in parsed_data.items():
            for item in items:
                record_id = uuid.uuid4().hex[:8]

                if isinstance(item, dict):
                    # Structured format with segments
                    text = item.get('text', '')
                    segments = item.get('segments', [])

                    rows.append({
                        'id': record_id,
                        'text': text,
                        'type': main_type,
                        'segments': segments if segments else None,
                        'has_segments': len(segments) > 0
                    })
                else:
                    # Simple string format
                    rows.append({
                        'id': record_id,
                        'text': str(item),
                        'type': main_type,
                        'segments': None,
                        'has_segments': False
                    })

        return pd.DataFrame(rows)

    def process(self, prompts: Dict[str, str]) -> pd.DataFrame:
        """Process prompts and return generated text DataFrame."""
        all_dataframes = []

        for prompt_type, prompt in prompts.items():
            try:
                generated_output = self.generate(prompt)
                parsed_data = self._parse_generated_json(generated_output)
                df = self.create_dataframe(parsed_data)
                all_dataframes.append(df)
            except Exception as e:
                print(f"Error generating for {prompt_type}: {e}")
                continue

        if all_dataframes:
            return pd.concat(all_dataframes, ignore_index=True)
        return pd.DataFrame()

    def _parse_generated_json(self, raw_output: str) -> Dict:
        """Parse generated JSON output."""
        # Clean JSON string
        cleaned = self._clean_json_string(raw_output)
        return json.loads(cleaned)

    def _clean_json_string(self, raw_output: str) -> str:
        """Clean JSON string by removing markdown formatting."""
        import re
        cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_output.strip())
        return cleaned

    def get_config(self) -> Dict:
        """Return processor configuration."""
        return self.config