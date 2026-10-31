import logging
import os
from logging.handlers import RotatingFileHandler

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(name, log_file="logs/app.log", console_level=logging.INFO):
    logger = logging.getLogger(name)

    # if this logger was already set up, return it as it is
    # this stops duplicate handlers when the function is called again
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # do not pass messages up to the root logger, so they are not printed twice
    logger.propagate = False

    folder = os.path.dirname(log_file)
    if folder:
        os.makedirs(folder, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT)

    # console shows info and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    # main log file keeps everything, and starts a new file after about 100 KB
    file_handler = RotatingFileHandler(log_file, maxBytes=100000, backupCount=3)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # separate file that only keeps errors
    error_handler = RotatingFileHandler("logs/errors.log", maxBytes=100000, backupCount=3)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

    return logger
