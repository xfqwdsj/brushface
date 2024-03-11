from abc import ABC, abstractmethod
from typing import List

import numpy as np

from brushface.data.face_area import FaceArea


# Notice that all facial detector models must be inherited from this class


class Detector(ABC):
    name: str

    @abstractmethod
    def detect_faces(self, img: np.ndarray) -> List[FaceArea]:
        """
        Interface for detect and align face

        Args:
            img (np.ndarray): Pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
                where each object contains:

            - facial_area (FacialAreaRegion): The facial area region represented
                as x, y, w, h, left_eye and right_eye
        """
        pass
