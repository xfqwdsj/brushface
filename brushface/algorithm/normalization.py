from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Normalizer(ABC):
    name: str
    _instance: Optional["Normalizer"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @abstractmethod
    def _normalize(self, img: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Normalizes an image.

        Args:
            img: The input image.

        Returns:
            The normalized image.
        """

        return self._normalize(img)

    def __str__(self):
        return self.name


class RescaleNormalization(Normalizer):
    name = "rescale"

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        return img * 255


class FacenetNormalization(RescaleNormalization):
    name = "facenet"

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        img = super()._normalize(img)
        mean, std = img.mean(), img.std()
        return (img - mean) / std


class Facenet2018Normalization(RescaleNormalization):
    name = "facenet2018"

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        img = super()._normalize(img)
        return img / 127.5 - 1


class VggFace1Normalization(RescaleNormalization):
    name = "vggface1"

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        img = super()._normalize(img)
        return img - [93.5940, 104.7624, 129.1863]


class VggFace2Normalization(RescaleNormalization):
    name = "vggface2"

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        img = super()._normalize(img)
        return img - [91.4953, 103.8827, 131.0912]


class ArcFaceNormalization(RescaleNormalization):
    name = "arcface"

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        img = super()._normalize(img)
        return (img - 127.5) / 128
