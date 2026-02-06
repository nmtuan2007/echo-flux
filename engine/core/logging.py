import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from engine.core.config import Config


_initialized = False


def setup_logging(config: Config) -> logging.Logger:
    global _initialized
    if _initialized:
        return logging.getLogger("echoflux")

    logger = logging.getLogger("echoflux")
    logger.setLevel(config.get("logging.level", "INFO"))
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = config.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"session_{session_name}.log"

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=config.get("logging.max_bytes", 10_485_760),
        backupCount=config.get("logging.backup_count", 5),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _initialized = True
    logger.info("Logging initialized — log file: %s", log_file)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"echoflux.{name}")
