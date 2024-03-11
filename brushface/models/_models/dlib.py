import bz2
import tempfile
from pathlib import Path
from typing import List

import numpy as np

from brushface.internal import Logger
from brushface.internal.file import download, random_name, weights_path
from brushface.models.abstract import Recognizer, Threshold

logger = Logger(__name__)


class DlibClient(Recognizer):
    """
    Dlib model class
    """

    def __init__(
        self,
        threshold: Threshold = Threshold(cosine=0.07, euclidean=0.6, euclidean_l2=0.4),
    ):
        self.model = DlibResNet()
        self.name = "Dlib"
        self.input_shape = (150, 150)
        self.output_shape = 128
        self.threshold = threshold

    def find_embeddings(self, img: np.ndarray) -> List[float]:
        """
        find embeddings with Dlib model - different than regular models
        Args:
            img (np.ndarray): pre-loaded image in BGR
        Returns
            embeddings (list): multi-dimensional vector
        """
        # return self.model.predict(img)[0].tolist()

        # extract_faces returns 4 dimensional images
        if len(img.shape) == 4:
            img = img[0]

        # bgr to rgb
        img = img[:, :, ::-1]  # bgr to rgb

        # img is in scale of [0, 1] but expected [0, 255]
        if img.max() <= 1:
            img = img * 255

        img = img.astype(np.uint8)

        img_representation = self.model.model.compute_face_descriptor(img)
        img_representation = np.array(img_representation)
        img_representation = np.expand_dims(img_representation, axis=0)
        return img_representation[0].tolist()


class DlibResNet:
    # noinspection HttpUrlsUsage
    remote_file = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
    weights_file = weights_path() / "dlib_face_recognition_resnet_model_v1.dat"

    def __init__(self):

        # this is not a must dependency. do not import it in the global level.
        try:
            import dlib
        except ModuleNotFoundError as e:
            raise ImportError(
                "Dlib is an optional dependency, ensure the library is installed. "
                "Please install using `pip install dlib`."
            ) from e

        # download pre-trained model if it does not exist
        if not self.weights_file.exists():
            file_path = Path(tempfile.gettempdir()) / random_name()
            output = download(
                self.remote_file,
                file_path,
                logger_delegate=logger,
                content_tag="weights",
            )

            zipfile = bz2.BZ2File(output)
            data = zipfile.read()
            with open(self.weights_file, "wb") as f:
                f.write(data)

        self.model = dlib.face_recognition_model_v1(str(self.weights_file))


class DlibMetaData:
    def __init__(self):
        self.input_shape = [[1, 150, 150, 3]]
