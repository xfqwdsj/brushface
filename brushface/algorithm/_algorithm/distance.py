from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

from .l2 import l2_normalize
from .list import list_to_np_array


class DistanceCalculator(ABC):
    name: str

    @abstractmethod
    def _calculate(
        self,
        source_representation: Union[np.ndarray, List],
        test_representation: Union[np.ndarray, List],
    ) -> np.float64:
        pass

    @staticmethod
    def from_name(name: str) -> "DistanceCalculator":
        for subclass in DistanceCalculator.__subclasses__():
            if subclass.name == name:
                return subclass()
        raise ValueError(f"Invalid distance calculator name: {name}")

    def __call__(
        self,
        source_representation: Union[np.ndarray, List],
        test_representation: Union[np.ndarray, List],
    ):
        return self._calculate(source_representation, test_representation)

    def __str__(self):
        return self.name


class CosineDistance(DistanceCalculator):
    name = "cosine"

    def _calculate(
        self,
        source_representation: Union[np.ndarray, List],
        test_representation: Union[np.ndarray, List],
    ):
        source_representation, test_representation = list_to_np_array(
            source_representation, test_representation
        )

        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


class EuclideanDistance(DistanceCalculator):
    name = "euclidean"

    def _calculate(
        self,
        source_representation: Union[np.ndarray, List],
        test_representation: Union[np.ndarray, List],
    ):
        source_representation, test_representation = list_to_np_array(
            source_representation, test_representation
        )

        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance


class EuclideanL2Distance(EuclideanDistance):
    name = "euclidean_l2"

    def _calculate(
        self,
        source_representation: Union[np.ndarray, List],
        test_representation: Union[np.ndarray, List],
    ):
        source_representation, test_representation = list_to_np_array(
            source_representation, test_representation
        )
        return super()(
            l2_normalize(source_representation), l2_normalize(test_representation)
        )
