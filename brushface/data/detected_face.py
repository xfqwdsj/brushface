from typing import TypedDict

import numpy as np

from brushface.data.face_area import FaceArea


class DetectedFace(TypedDict):
    """
    A dictionary containing an image and the rectangular region of a face.

    Attributes:
        img: The image containing the detected face.
        face_area: The rectangular region of the face in the image.
    """

    img: np.ndarray
    face_area: FaceArea
