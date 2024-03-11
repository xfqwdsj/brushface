import typing

import numpy as np

from brushface.internal.logger import Logger
from brushface.models.abstract.analyzer import (
    IntegerResult,
    VggFaceBasedAnalyzer,
)

logger = Logger(__name__)


class ApparentAgeClient(VggFaceBasedAnalyzer):
    name = "Age"

    def __init__(self):
        self.model = VggFaceBasedAnalyzer.build_model(
            weights_url="https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5",
            classes=101,
            logger_delegate=logger,
        )

    def analyze(self, img: np.ndarray) -> IntegerResult:
        result = self.find_apparent_age(self.predict(img))
        return {
            "result": int(result),
            "raw_value": float(result),
        }

    @staticmethod
    def find_apparent_age(age_predictions: np.ndarray) -> np.float64:
        """Finds apparent age prediction from a given probes of ages."""

        output_indexes = np.array(list(range(0, 101)))
        apparent_age = typing.cast(np.float64, np.sum(age_predictions * output_indexes))
        return apparent_age
