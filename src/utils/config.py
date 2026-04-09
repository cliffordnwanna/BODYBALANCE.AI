"""
BODYBALANCE.AI - Configuration Management
Loads configuration from YAML file and environment variables.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    "similarity_threshold": 0.3,
    "google_sheet_id": "",
    "google_credentials_path": "",
    "cache_ttl_minutes": 5,
    "session_timeout_minutes": 30,
    "feedback_enabled": True,
    "admin_password": "admin123",
    "qa_text_path": "training_data.txt",
    "qa_json_path": "qa_data.json",
    "log_level": "INFO",
    "app_title": "BODYBALANCE.AI",
    "app_subtitle": "Your AI-Powered Wellness Assistant",
    "primary_color": "#2E7D32",
    "background_color": "#F5F5F5",
    "support_email": "support@bodybalance.com",
    "support_phone": "+234 XXX XXX XXXX"
}


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file and environment variables.
    
    Priority (highest to lowest):
    1. Environment variables (prefixed with BODYBALANCE_)
    2. config.yaml file
    3. Default values
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    # Load from YAML file if exists
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f) or {}
            config.update(yaml_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load {config_path}: {e}")
    
    # Override with environment variables
    env_mappings = {
        "BODYBALANCE_SIMILARITY_THRESHOLD": ("similarity_threshold", float),
        "BODYBALANCE_GOOGLE_SHEET_ID": ("google_sheet_id", str),
        "GOOGLE_APPLICATION_CREDENTIALS": ("google_credentials_path", str),
        "BODYBALANCE_CACHE_TTL": ("cache_ttl_minutes", int),
        "BODYBALANCE_SESSION_TIMEOUT": ("session_timeout_minutes", int),
        "BODYBALANCE_FEEDBACK_ENABLED": ("feedback_enabled", _parse_bool),
        "BODYBALANCE_ADMIN_PASSWORD": ("admin_password", str),
        "BODYBALANCE_LOG_LEVEL": ("log_level", str),
        "BODYBALANCE_SUPPORT_EMAIL": ("support_email", str),
        "BODYBALANCE_SUPPORT_PHONE": ("support_phone", str),
    }
    
    for env_var, (config_key, converter) in env_mappings.items():
        env_value = os.environ.get(env_var)
        if env_value is not None:
            try:
                config[config_key] = converter(env_value)
                logger.debug(f"Config override from env: {config_key}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid env value for {env_var}: {e}")
    
    return config


def _parse_bool(value: str) -> bool:
    """Parse a string to boolean."""
    return value.lower() in ("true", "1", "yes", "on")


def save_config(config: Dict[str, Any], config_path: str = "config.yaml"):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to save the YAML file
    """
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")


def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from src/utils to project root
    current = Path(__file__).resolve()
    return current.parent.parent.parent


# Singleton config instance
_config: Optional[Dict[str, Any]] = None


def get_config() -> Dict[str, Any]:
    """
    Get the global configuration instance.
    
    Returns:
        Configuration dictionary
    """
    global _config
    if _config is None:
        # Try to find config.yaml in project root
        root = get_project_root()
        config_path = root / "config.yaml"
        _config = load_config(str(config_path))
    return _config


def reload_config():
    """Reload the configuration from file."""
    global _config
    _config = None
    return get_config()
