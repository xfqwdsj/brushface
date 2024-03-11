from abc import ABC, abstractmethod
from typing import Dict, Generic, List, TypeVar, TypedDict

import numpy as np
from keras.layers import Convolution2D
from keras.models import Model
from keras.src.layers import Activation, Flatten

from brushface.internal.file import get_weights
from brushface.internal.logger import Logger
from brushface.models.abstract.model import BrushFaceModel
from brushface.models.vgg_face import VggFaceClient


RESULT_TYPE = TypeVar("RESULT_TYPE")


class AnalyzerResult(TypedDict, Generic[RESULT_TYPE]):
    result: RESULT_TYPE


class LabeledResult(AnalyzerResult):
    possibilities: Dict[RESULT_TYPE, float]


class IntegerResult(AnalyzerResult):
    raw_value: float


class Analyzer(BrushFaceModel, ABC):
    model: Model

    @abstractmethod
    def analyze(self, img: np.ndarray) -> AnalyzerResult:
        pass


class VggFaceBasedAnalyzer(Analyzer, ABC):
    @staticmethod
    def build_model(weights_url: str, classes: int, logger_delegate: Logger) -> Model:
        vgg_face_model = VggFaceClient.base_model()

        outputs = Convolution2D(classes, (1, 1), name="predictions")(
            vgg_face_model.layers[-4].output
        )
        outputs = Flatten()(outputs)
        outputs = Activation("softmax")(outputs)

        model = Model(inputs=vgg_face_model.inputs, outputs=outputs)
        model.load_weights(get_weights(weights_url, logger_delegate))

        return model

    def predict(self, img: np.ndarray) -> np.ndarray:
        return self.model.predict(img, verbose=0)[0, :]


class LabeledAnalyzer(Analyzer, ABC):
    labels: List[str]

    def get_result(self, predictions: np.ndarray) -> LabeledResult:
        result: LabeledResult = {
            "result": self.labels[np.argmax(predictions)],
            "possibilities": {},
        }

        for i, label in enumerate(self.labels):
            prediction = predictions[i]
            result["possibilities"][label] = float(prediction)

        return result
