from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type

import numpy as np
from keras.models import Model

from brushface.algorithm.distance import DistanceCalculator
from brushface.models.abstract.model import BrushFaceModel

type Threshold = Dict[Type[DistanceCalculator], float]


class Recognizer(BrushFaceModel, ABC):
    input_shape: Tuple[int, int]
    output_shape: int
    threshold: Threshold

    @abstractmethod
    def find_embeddings(self, img: np.ndarray) -> List[float]:
        """
        Finds embeddings with the model.

        Args:
            img: Image in BGR format.

        Returns:
            Multi-dimensional vector.
        """

        pass


class KerasRecognizer(Recognizer):
    model: Model

    def find_embeddings(self, img: np.ndarray):
        result = self.model(img, training=False)

        try:
            result = result.numpy()
        except TypeError:
            result = result.cpu().detach().numpy()

        return result[0].tolist()
