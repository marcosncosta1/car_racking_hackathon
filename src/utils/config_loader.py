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
    # Check for either nested format (after adaptation) or flat format (original)
    # Nested format: 'env', 'preprocessing', 'network', 'agent', 'training'
    # Flat format: 'env_name', 'preprocessing', 'network_architecture', 'dqn', 'training'

    is_nested = 'env' in config and 'agent' in config
    is_flat = 'env_name' in config and 'dqn' in config

    if not (is_nested or is_flat):
        raise ValueError("Config must have either nested format (env, agent) or flat format (env_name, dqn)")

    # Validate flat format (original)
    if is_flat:
        required_flat = ['env_name', 'preprocessing', 'network_architecture', 'dqn', 'training']
        for section in required_flat:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")

    # Validate nested format (adapted)
    if is_nested:
        required_nested = ['env', 'preprocessing', 'network', 'agent', 'training']
        for section in required_nested:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")


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
