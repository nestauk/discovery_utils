"""Utility functions for input/output operations"""

from typing import Dict

import yaml


def safe_yaml_load(yaml_str: str) -> Dict:
    """Safely load a YAML string"""
    try:
        if type(yaml_str) is str:
            return yaml.safe_load(open(yaml_str))
        else:
            return yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {e}")


def remap_dict(category_dict: dict) -> dict:
    """Remaps a nested dictionary so that each item maps to its corresponding category."""
    return {item: category for category, items in category_dict.items() for item in items}
