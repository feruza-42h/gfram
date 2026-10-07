"""
Configuration management for GFRAM.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for GFRAM."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.

        Args:
            config_dict: Configuration dictionary.
        """
        self.config = config_dict or self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'detector': {
                'max_num_faces': 1,
                'min_detection_confidence': 0.5,
                'refine_landmarks': True,
            },
            'features': {
                'extract_euclidean': True,
                'extract_differential': True,
                'extract_topological': True,
                'extract_statistical': True,
                'extract_symmetry': True,
                'extract_graph': True,
            },
            'model': {
                'name': 'geometric_transformer',
                'config': 'base',
                'embed_dim': 256,
                'num_layers': 8,
            },
            'index': {
                'dimension': 256,
                'index_type': 'HNSW',
                'metric': 'cosine',
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 100,
                'optimizer': 'adam',
            }
        }

    @classmethod
    def from_file(cls, path: str) -> 'Config':
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            Config instance.
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(config_dict)

    def save(self, path: str):
        """
        Save configuration to YAML file.

        Args:
            path: Path to save file.
        """
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value by key."""
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def __repr__(self) -> str:
        return f"Config({self.config})"