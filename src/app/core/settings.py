import logging
import os

from starlette.datastructures import CommaSeparatedStrings

WANT_RETRAINING = os.getenv("WANT_RETRAINING", False)
API_PORT = 8080
ROOT_PATH = os.getenv("ROOT_PATH", "")
API_DEBUG = True
ALLOWED_HOSTS = CommaSeparatedStrings(os.getenv("ALLOWED_HOSTS", "*"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "debug")
OPEN_API_VERSION = "2.1.0"
logger = logging.getLogger(f"{os.getenv('LOGGER', 'gunicorn')}.error")
