import random
import string
from pathlib import Path
from typing import Union

import gdown

from brushface.internal.env import Env
from brushface.internal.logger import Logger

logger = Logger(__name__)


def initialize():
    """
    Initializes the directory for storing model weights.

    Raises:
        OSError: if the directory cannot be created.
    """

    for path in [lib_home_path(), lib_weights_path()]:
        if not path.exists():
            path.mkdir(exist_ok=True)
            logger.info(f"Directory {path} created.")


def lib_home_path():
    return Env.home()[0]


def lib_weights_path():
    return lib_home_path() / "weights"


def download(
    url: str,
    output: Union[Path, str],
    logger_delegate=logger,
    content_description="file",
):
    """
    Downloads file from URL to output.

    Args:
        url: URL of the file.
        output: Path to save the file
        logger_delegate: The logger to use.
        content_description: Description of the content to download.

    Returns:
        Output file path.
    """

    logger_delegate.info(f"Downloading {content_description} from {url} ...")
    if isinstance(output, Path):
        output = str(output.resolve())
    return gdown.download(url, output, resume=True)


def get_weights(url: str, logger_delegate=logger):
    """
    Downloads weights from URL if not exists and return the path.

    Args:
        url: URL of the weights file.
        logger_delegate: The logger to use.

    Returns:
        path to the weights file
    """

    output = (lib_weights_path() / url.split("/")[-1]).resolve()

    if not output.exists():
        download(
            url,
            str(output),
            logger_delegate=logger_delegate,
            content_description="weights",
        )

    return output


def random_name():
    """
    Generates a random file name for temporary files.

    Returns:
        random file name
    """

    return "".join(random.choices(string.ascii_letters + string.digits, k=8))
