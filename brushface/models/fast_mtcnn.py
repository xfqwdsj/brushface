from typing import Any, List, Optional, Union

import numpy as np

from brushface.data.face_area import FaceArea
from brushface.error.optional_dependency_not_found_error import (
    OptionalDependencyNotFoundError,
)
from brushface.models.abstract.detector import Detector


# Link -> https://github.com/timesler/facenet-pytorch
# Examples https://www.kaggle.com/timesler/guide-to-mtcnn-in-facenet-pytorch


class FastMtcnnClient(Detector):
    name = "FastMtcnn"

    def __init__(self, model: Optional[Any] = None):
        if model is not None:
            self.model = model
            return

        self.model = self.build_model()

    def detect_faces(self, img: np.ndarray) -> List[FaceArea]:
        """
        Detect and align face with mtcnn

        Args:
            img (np.ndarray): pre-loaded image as NumPy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        resp = []

        img_rgb = img[:, :, ::-1]  # mtcnn expects RGB
        detections = self.model.detect(
            img_rgb, landmarks=True
        )  # returns boundingbox, prob, landmark
        if (
            detections is not None
            and len(detections) > 0
            and not any(detection is None for detection in detections)
        ):  # issue 1043
            for current_detection in zip(*detections):
                x, y, w, h = xyxy_to_xywh(current_detection[0])
                confidence = current_detection[1]

                left_eye = current_detection[2][0]
                right_eye = current_detection[2][1]

                resp.append(
                    {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "left_eye": (int(left_eye[0]), int(left_eye[1])),
                        "right_eye": (int(right_eye[0]), int(right_eye[1])),
                        "confidence": confidence,
                    }
                )

        return resp

    @staticmethod
    def build_model(**kwargs):
        """
        Build a fast mtcnn face detector model
        Returns:
            model (Any)
        """

        try:
            from facenet_pytorch import MTCNN
        except ModuleNotFoundError as e:
            raise OptionalDependencyNotFoundError("facenet-pytorch", "fastmtcnn") from e

        face_detector = MTCNN(
            image_size=160,
            thresholds=[0.6, 0.7, 0.7],
            post_process=True,
            select_largest=False,
            **kwargs
        )  # return result in descending order
        return face_detector


def xyxy_to_xywh(xyxy: Union[list, tuple]) -> list:
    """
    Convert xyxy format to xywh format.
    """
    x, y = xyxy[0], xyxy[1]
    w = xyxy[2] - x + 1
    h = xyxy[3] - y + 1
    return [x, y, w, h]
