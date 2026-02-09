"""Configuration loading utilities.

Loads and validates YAML configuration files.
"""

import yaml
from pathlib import Path


def load_config(config_path):
    """Load configuration from YAML file.

    Args:
        config_path (str): Path to config file

    Returns:
        dict: Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    _validate_config(config)

    return config


def _validate_config(config):
    """Validate configuration has required fields.

    Args:
        config (dict): Configuration dictionary

    Raises:
        ValueError: If required fields are missing
    """
    required_sections = ['env', 'preprocessing', 'network', 'agent', 'training']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate specific fields
    if 'num_actions' not in config['env']:
        raise ValueError("Missing 'num_actions' in env config")

    if 'gamma' not in config['agent']:
        raise ValueError("Missing 'gamma' in agent config")


def save_config(config, save_path):
    """Save configuration to YAML file.

    Args:
        config (dict): Configuration dictionary
        save_path (str): Path to save config
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Config saved to {save_path}")
