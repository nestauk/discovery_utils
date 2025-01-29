"""Utility functions for input/output operations"""

from typing import Dict

import yaml


def safe_yaml_load(yaml_str: str) -> Dict:
    """Safely load a YAML string"""
    try:
        return yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {e}")
