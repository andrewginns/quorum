"""Configuration handling for Quorum proxy."""

import yaml
import logging
import os
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

aggregation_logger = logging.getLogger("aggregation")
aggregation_logger.setLevel(logging.INFO)

log_dir = Path(__file__).parent.parent.parent / "logs"
os.makedirs(log_dir, exist_ok=True)

aggregation_log_file = log_dir / "aggregation.log"
file_handler = logging.FileHandler(str(aggregation_log_file), mode="a")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)

aggregation_logger.addHandler(file_handler)
aggregation_logger.propagate = True

try:
    with open(str(aggregation_log_file), "a") as f:
        f.write("Test direct write to log file\n")
    logger.info(f"Successfully wrote to log file at {aggregation_log_file}")
except Exception as e:
    logger.error(f"Failed to write to log file: {str(e)}")


def load_config():
    """
    Load configuration from config.yaml file.
    Returns a dictionary containing the configuration.
    """
    try:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        config_yaml = config_path.read_text()
        config = yaml.safe_load(config_yaml)
        logger.info("Successfully loaded configuration from config.yaml")
        return config
    except Exception as e:
        logger.error(f"Error loading config.yaml: {str(e)}")
        return {
            "primary_backends": [
                {
                    "name": "default",
                    "url": "https://api.openai.com/v1",
                    "model": "",
                }
            ],
            "settings": {"timeout": 60},
        }


config = load_config()

target_backend = config["primary_backends"][0]
OPENAI_API_BASE = target_backend["url"]
DEFAULT_MODEL = target_backend.get("model", "")

if not OPENAI_API_BASE:
    logger.warning("Backend URL not set in config.yaml, using default value")
    OPENAI_API_BASE = "https://api.openai.com/v1"

TIMEOUT = config["settings"].get("timeout", 60)
