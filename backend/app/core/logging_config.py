import logging
import os
from logging.handlers import RotatingFileHandler

from .config import settings


def setup_logging():
    log_file = os.path.join(settings.LOGS_DIR, "app.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5),
            logging.StreamHandler(),
        ],
    )
    logging.info("Logging setup complete.")
