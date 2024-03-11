from typing import Any, List, Optional

import numpy as np

from brushface.data.face_area import FaceArea
from brushface.internal import Logger
from brushface.internal.file import download, weights_path
from brushface.models.abstract import Detector

logger = Logger(__name__)


class YoloClient(Detector):
    name = "Yolo"

    def __init__(self, model: Optional[Any] = None):
        if model is not None:
            self.model = model
            return

        self.model = self.build_model()

    @staticmethod
    def build_model():
        """
        Build a yolo detector model
        Returns:
            model (Any)
        """

        # Import the Ultralytics YOLO model
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as e:
            raise ImportError(
                "Yolo is an optional detector. Please install using `pip install ultralytics` "
            ) from e

        # Google Drive URL from repo (https://github.com/derronqi/yolov8-face) ~6MB
        url = "https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb"
        file_name = "yolov8n-face.pt"

        file_path = weights_path() / file_name

        # Download the model's weights if they don't exist
        if not file_path.is_file():
            download(url, file_path)

        # Return face_detector
        return YOLO(file_path)

    def detect_faces(self, img: np.ndarray) -> List[FaceArea]:
        """
        Detect and align face with yolo

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        resp = []

        # Detect faces
        from ultralytics.engine.results import Results

        results: List[Results] = self.model.predict(
            img, verbose=False, show=False, conf=0.25
        )[0]

        # For each face, extract the bounding box, the landmarks and confidence
        for result in results:

            if result.boxes is None or result.keypoints is None:
                continue

            # Extract the bounding box and the confidence
            x, y, w, h = result.boxes.xywh.tolist()[0]
            confidence = result.boxes.conf.tolist()[0]

            # left_eye_conf = result.keypoints.conf[0][0]
            # right_eye_conf = result.keypoints.conf[0][1]
            left_eye = result.keypoints.xy[0][0].tolist()
            right_eye = result.keypoints.xy[0][1].tolist()

            x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)

            resp.append(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "confidence": confidence,
                    "left_eye": (int(left_eye[0]), int(left_eye[1])),
                    "right_eye": (int(right_eye[0]), int(right_eye[1])),
                }
            )

        return resp
