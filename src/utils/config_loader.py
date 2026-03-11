"""
Configuration Loader for EJB VLM
Author: Eduardo J. Barrios (@edujbarrIos)
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


_config_cache = None


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to config file. If None, uses default.
        
    Returns:
        dict: Configuration dictionary
    """
    global _config_cache
    
    if config_path is None:
        # Find config relative to this file
        current_dir = Path(__file__).parent.parent.parent
        config_path = current_dir / "config" / "config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    _config_cache = config
    return config


def get_config(reload: bool = False) -> Dict[str, Any]:
    """
    Get cached configuration or load if not already loaded.
    
    Args:
        reload (bool): Force reload from file
        
    Returns:
        dict: Configuration dictionary
    """
    global _config_cache
    
    if _config_cache is None or reload:
        return load_config()
    
    return _config_cache


def get_model_config(model_type: str = "default") -> Dict[str, Any]:
    """
    Get model-specific configuration.
    
    Args:
        model_type (str): Model type (default, creative, focused, detailed)
        
    Returns:
        dict: Model configuration
    """
    config = get_config()
    
    model_cfg = {
        "clip": config["models"]["clip"],
        "gpt": config["models"]["gpt"],
        "generation": config["generation"].get(model_type, config["generation"]["default"])
    }
    
    return model_cfg


def get_categories(category_set: str = "default") -> list:
    """
    Get category list for zero-shot classification.
    
    Args:
        category_set (str): Category set name (default or extended)
        
    Returns:
        list: List of categories
    """
    config = get_config()
    return config["categories"].get(category_set, config["categories"]["default"])


def get_templates(template_set: str = "basic") -> list:
    """
    Get image context templates.
    
    Args:
        template_set (str): Template set name (basic or extended)
        
    Returns:
        list: List of templates
    """
    config = get_config()
    return config["templates"].get(template_set, config["templates"]["basic"])


def get_generation_params(preset: str = "default") -> Dict[str, Any]:
    """
    Get generation parameters for a specific preset.
    
    Args:
        preset (str): Preset name (default, creative, focused, detailed)
        
    Returns:
        dict: Generation parameters
    """
    config = get_config()
    return config["generation"].get(preset, config["generation"]["default"])


def get_paths() -> Dict[str, str]:
    """
    Get all configured paths.
    
    Returns:
        dict: Path configuration
    """
    config = get_config()
    return config["paths"]


def create_paths():
    """Create all configured directories if they don't exist."""
    paths = get_paths()
    base_dir = Path(__file__).parent.parent.parent
    
    for path_name, path_value in paths.items():
        full_path = base_dir / path_value
        full_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test config loading
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"\nProject: {config['project']['name']}")
    print(f"Author: {config['project']['author']}")
    print(f"\nCLIP Model: {config['models']['clip']['model_name']}")
    print(f"GPT Model: {config['models']['gpt']['model_name']}")
    print(f"\nDefault Categories: {len(config['categories']['default'])} items")
