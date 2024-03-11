import random
import string
from pathlib import Path
from typing import Union

import gdown

from brushface.internal import Env, Logger

logger = Logger(__name__)


def initialize():
    """
    Initialize the folder for storing model weights.

    Raises:
        OSError: if the folder cannot be created.
    """

    for path in [home_path(), weights_path()]:
        if not path.exists():
            path.mkdir(exist_ok=True)
            logger.info(f"Directory {path} created.")


def home_path():
    return Env.home()[0]


def weights_path():
    return home_path() / "weights"


def download(
    url: str, output: Union[Path, str], logger_delegate=logger, content_tag="file"
):
    """
    Download file from URL to output.

    Args:
        url: URL of the file.
        output: Path to save the file
        logger_delegate: The logger to use.
        content_tag: The tag to use in the log message.
    """

    logger_delegate.info(f"Downloading {content_tag} from {url} ...")
    if isinstance(output, Path):
        output = str(output.resolve())
    return gdown.download(url, output, quiet=False)


def weights(url: str, logger_delegate=logger):
    """
    Download weights from URL if not exists and return the path.

    Args:
        url: URL of the weights file.
        logger_delegate: The logger to use.

    Returns:
        path to the weights file
    """

    output = (weights_path() / url.split("/")[-1]).resolve()

    if not output.exists():
        download(
            url, str(output), logger_delegate=logger_delegate, content_tag="weights"
        )

    return output


def random_name():
    """
    Generate a random file name for temporary files.

    Returns:
        random file name
    """

    return "".join(random.choices(string.ascii_letters + string.digits, k=8))
