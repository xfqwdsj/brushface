from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union

import numpy as np
from keras.models import Model


# Notice that all facial recognition models must be inherited from this class


class Threshold:
    cosine: float
    euclidean: float
    euclidean_l2: float

    def __init__(
        self, cosine: float = 0.4, euclidean: float = 0.55, euclidean_l2: float = 0.75
    ):
        self.cosine = cosine
        self.euclidean = euclidean
        self.euclidean_l2 = euclidean_l2


class Recognizer(ABC):
    model: Union[Model, Any]
    name: str
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

    def __str__(self):
        return self.name


class KerasRecognizer(Recognizer):
    def find_embeddings(self, img: np.ndarray) -> List[float]:
        result = self.model(img, training=False)

        try:
            result = result.numpy()
        except TypeError:
            result = result.cpu().detach().numpy()

        return result[0].tolist()
