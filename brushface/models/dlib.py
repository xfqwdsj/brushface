import bz2
import tempfile
from pathlib import Path
from types import MappingProxyType

import cv2
import numpy as np

from brushface.algorithm.distance import (
    CosineDistance,
    EuclideanDistance,
    EuclideanL2Distance,
)
from brushface.error.optional_dependency_not_found_error import (
    OptionalDependencyNotFoundError,
)
from brushface.internal.file import download, lib_weights_path
from brushface.internal.logger import Logger
from brushface.models.abstract.recognizer import Recognizer, Threshold

logger = Logger(__name__)


class DlibClient(Recognizer):
    name = "Dlib"
    input_shape = (150, 150)
    output_shape = 128

    def __init__(
        self,
        threshold: Threshold = MappingProxyType(
            {
                CosineDistance: 0.07,
                EuclideanDistance: 0.6,
                EuclideanL2Distance: 0.4,
            }
        ),
    ):
        self.model = self.build_model()
        self.threshold = threshold

    def find_embeddings(self, img: np.ndarray):
        if len(img.shape) == 4:
            img = img[0]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.max() <= 1:
            img = img * 255

        img = img.astype(np.uint8)

        img_representation = self.model.compute_face_descriptor(img)
        img_representation = np.array(img_representation)
        img_representation = np.expand_dims(img_representation, axis=0)
        return img_representation[0].tolist()

    @staticmethod
    def build_model(
        bz2_url="http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
    ):
        try:
            import dlib
        except ModuleNotFoundError as e:
            raise OptionalDependencyNotFoundError("dlib", "dlib") from e

        bz2_file_name = bz2_url.split("/")[-1]
        weights_name = bz2_file_name.replace(".bz2", "")

        weights_path = lib_weights_path() / weights_name

        if not weights_path.exists():
            file_path = Path(tempfile.gettempdir()) / bz2_file_name
            output = download(
                bz2_url,
                file_path,
                logger_delegate=logger,
                content_description="weights",
            )

            zipfile = bz2.BZ2File(output)
            data = zipfile.read()
            with open(weights_path, "wb") as f:
                f.write(data)

        return dlib.face_recognition_model_v1(str(weights_path))
