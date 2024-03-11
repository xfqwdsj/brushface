import logging
import os
from pathlib import Path


class Env:
    @staticmethod
    def home():
        key = "BRUSHFACE_HOME"
        env = os.getenv(key)
        if env:
            path = Path(env)
        else:
            path = Path.home() / ".brushface"
        return path, key

    @staticmethod
    def log_level():
        key = "BRUSHFACE_LOG_LEVEL"
        return os.getenv(key, str(logging.INFO)), key
