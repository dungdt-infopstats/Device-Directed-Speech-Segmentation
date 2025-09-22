"""
Text processing utilities for extracting active text labels.
"""

import pandas as pd
from typing import Union, Dict, Any


def get_active_text_labels(phase2_data: Union[pd.DataFrame, Dict], file_id: str) -> str:
    """
    Extract and concatenate active text from Phase 2 data.

    Args:
        phase2_data: DataFrame or dict from Phase 2 with text_segments
        file_id: ID of the file to process

    Returns:
        str: Concatenated active text segments
    """
    # Find the record for this file_id
    if isinstance(phase2_data, pd.DataFrame):
        matching_rows = phase2_data[phase2_data['id'] == file_id]
        if matching_rows.empty:
            return ""
        record = matching_rows.iloc[0]
    else:
        record = phase2_data

    # If no segments, return the full command for single_active/chain_active
    if record.get('num_segments', 0) == 0:
        if record.get('type') in ['single_active', 'chain_active']:
            return record.get('command', '')
        else:
            return ""

    # Extract active segments from text_segments and type_segments
    text_segments = record.get('text_segments', {})
    type_segments = record.get('type_segments', {})

    active_texts = []
    for seg_id, seg_type in type_segments.items():
        if seg_type == 'active':
            active_text = text_segments.get(seg_id, '')
            if active_text:
                active_texts.append(active_text.strip())

    return ' '.join(active_texts)


def extract_file_id_from_path(file_name: str, separator: str = '_') -> str:
    """
    Extract file ID from file name pattern like 'type_id'.

    Args:
        file_name: File name in format 'type_id'
        separator: Separator character

    Returns:
        str: Extracted file ID
    """
    parts = file_name.split(separator, 1)
    return parts[1] if len(parts) > 1 else file_name