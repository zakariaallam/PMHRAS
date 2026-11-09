import logging
from pathlib import Path
from typing import Any

import yaml

from pmhras.data.exceptions import (AuditConfigurationError)

logger = logging.getLogger(__name__)

def load_config(config_path: str | Path) -> dict[str,Any]:

    path = Path(config_path)

    if not path.exists():
        raise AuditConfigurationError(
            f"Configuration file does not exist {path}"
        )

    try:
        with path.open("r",encoding="utf-8") as file:
            config = yaml.safe_load(file)

    except yaml.YAMLError as exc:
        raise AuditConfigurationError(
            f"Invalide yaml configuration  {path}"
            ) from exc

    if not isinstance(config,dict):
        raise AuditConfigurationError(
            "Configuration must contain a YAML mapping"
        )

    required_sections = {"data", "audit", "report", "logging"}

    missing_sections = required_sections.difference(config)

    if missing_sections:
        raise AuditConfigurationError(
            f"Missing configuration sections: {sorted(missing_sections)}"
        )

    data_config = config["data"]

    if not isinstance(data_config, dict):
        raise AuditConfigurationError(
            "'data' configuration must be a mapping."
        )

    if not data_config.get("raw_path"):
        raise AuditConfigurationError(
            "'data.raw_path' is required."
        )

    return config 