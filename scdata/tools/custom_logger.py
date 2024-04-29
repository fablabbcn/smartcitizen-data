from termcolor import colored
from scdata._config import config
from datetime import datetime
import sys
import logging

class CutsomLoggingFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_min = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    format_deb = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format_min + reset,
        logging.INFO: grey + format_min + reset,
        logging.WARNING: yellow + format_min + reset,
        logging.ERROR: red + format_deb + reset,
        logging.CRITICAL: bold_red + format_deb + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logger = logging.getLogger('scdata')
logger.setLevel(config._log_level)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(config._log_level)
ch.setFormatter(CutsomLoggingFormatter())
logger.addHandler(ch)

def set_logger_level(level=logging.DEBUG):
    logger.setLevel(level)
