"""Utility functions for input/output operations"""

from typing import Dict, List

import yaml


def safe_yaml_load(yaml_str: str) -> Dict:
    """Safely load a YAML string"""
    try:
        return yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {e}")


def remap_dict(one_to_many: Dict[str, List[str]]) -> Dict[str, str]:
    """Transform the one-to-many mapping to many-to-one mapping."""
    original_mapping = {}
    for one, many in one_to_many.items():
        for y in many:
            original_mapping[y] = one
    return original_mapping
    