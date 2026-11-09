import logging
import logging.config
from pathlib import Path

def configure_logging(
    log_dir: str = "logs",
    filename: str = "data_audit.log",
    level: str = "INFO",
    ) -> None:

    log_path = Path(log_dir)
    log_path.mkdir(parents=True,exist_ok=True)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": (
                        "%(asctime)s | %(levelname)s | "
                        "%(name)s | %(message)s"
                    ),
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": level.upper(),
                    "formatter": "standard",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": str(log_path / filename),
                    "maxBytes": 5_000_000,
                    "backupCount": 3,
                    "encoding": "utf-8",
                    "level": level.upper(),
                    "formatter": "standard",
                },
            },
            "root": {
                "level": level.upper(),
                "handlers": ["console", "file"],
            },
        }
    )
