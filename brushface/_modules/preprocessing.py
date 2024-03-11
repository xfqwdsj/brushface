import base64
from pathlib import Path

import cv2
import numpy as np
import requests

from brushface._modules.type import _Img


def load_image(img: _Img):
    """
    Loads image from path, URL, base64 or numpy array.

    Args:
        img: A path, URL, base64 or numpy array.

    Returns:
        The loaded image in BGR format and image name itself.
    """

    if isinstance(img, np.ndarray):
        return img, "numpy array"

    if isinstance(img, str):
        if img.startswith("data:image/"):
            return load_base64(img), "base64 encoded string"

        if img.lower().startswith("http://") or img.lower().startswith("https://"):
            return load_image_from_web(url=img), img

        img = Path(img)

    if not img.is_file():
        raise ValueError("Cannot read the image.")

    img_path = str(img)

    return cv2.imread(img_path), img_path


def load_image_from_web(url: str):
    """
    Loads an image from web.

    Args:
        url: link to the image.

    Returns:
        The loaded image in BGR format.
    """

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    image_array = np.asarray(bytearray(response.raw.read()), dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def load_base64(uri: str):
    """
    Loads image from base64 encoded string.

    Args:
        uri: A base64 encoded string.

    Returns:
        The loaded image in BGR format.
    """

    encoded_data_parts = uri.split(",")

    if len(encoded_data_parts) < 2:
        raise ValueError("Format error in base64 encoded string.")

    # similar to find functionality, we are just considering these extensions
    if not (
        uri.startswith("data:image/jpeg")
        or uri.startswith("data:image/jpg")
        or uri.startswith("data:image/png")
    ):
        raise ValueError(
            f"Expected image format is jpeg, jpg or png but got {uri.split(';')[0].split('/')[1]}."
        )

    encoded_data = encoded_data_parts[1]

    array = np.frombuffer(base64.b64decode(encoded_data), dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)
