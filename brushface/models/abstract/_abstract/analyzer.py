from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from keras.models import Model


# Notice that all facial attribute analysis models must be inherited from this class


class Analyzer(ABC):
    model: Model
    name: str

    @abstractmethod
    def predict(self, img: np.ndarray) -> Union[np.ndarray, np.float64]:
        pass

    def __str__(self):
        return self.name
