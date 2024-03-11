from typing import List, Optional

import cv2
import numpy as np

from brushface._modules import detection, preprocessing
from brushface._modules.type import _Detector, _Img, _Recognizer, extract_type
from brushface.algorithm import default_normalizer
from brushface.data.detected_face import DetectedFace
from brushface.data.representation import Representation
from brushface.models import default_detector, default_recognizer


def represent(
    img: _Img,
    normalizer=default_normalizer,
    detector: Optional[_Detector] = default_detector,
    enforce_detection=True,
    align=True,
    expand_percentage=0,
    recognizer: _Recognizer = default_recognizer,
) -> List[Representation]:
    """
    Represents face images as multidimensional vector embeddings.

    Args:
        img: The path or URL to the image, a numpy array in BGR format, or a base64 encoded image.
        normalizer: The normalizer to use for preprocessing the image.
        detector: The face detector to use.
        enforce_detection: If no face is detected in an image, raise an exception.
            Not enforcing detection can be useful for low-resolution images.
        align: Perform alignment based on the eye positions.
        expand_percentage: The percentage to expand the detected face area.
        recognizer: The face recognizer to use.

    Returns:
        A list of representations containing the embeddings and face areas.
    """

    recognizer = extract_type(recognizer)

    resp_objs = []
    img_objs: List[DetectedFace]

    # we have run pre-process in verification. so, this can be skipped if it is coming from verify.
    target_size = recognizer.input_shape
    if detector is not None:
        img_objs = detection.extract_faces(
            img=img,
            target_size=(target_size[1], target_size[0]),
            grayscale=False,
            detector=detector,
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage,
            return_rgb=False,
        )
    else:  # skip
        # Try load. If load error, will raise exception internal
        img, _ = preprocessing.load_image(img)

        if len(img.shape) == 4:
            img = img[0]  # e.g. (1, 224, 224, 3) to (224, 224, 3)
        if len(img.shape) == 3:
            img = cv2.resize(img, target_size)
            img = np.expand_dims(img, axis=0)
            # when called from verify, this is already normalized. But needed when user given.
            if img.max() > 1:
                img = (img.astype(np.float32) / 255.0).astype(np.float32)

        # make dummy region and confidence to keep compatibility with `extract_faces`
        img_objs = [
            {
                "img": img,
                "face_area": {
                    "x": 0,
                    "y": 0,
                    "w": img.shape[1],
                    "h": img.shape[2],
                    "confidence": 0,
                    "left_eye": None,
                    "right_eye": None,
                },
            }
        ]

    for img_obj in img_objs:
        img = img_obj["img"]
        # custom normalization
        img = normalizer(img)

        embedding = recognizer.find_embeddings(img)

        resp_objs.append(
            {
                "embedding": embedding,
                "face_area": img_obj["face_area"],
            }
        )

    return resp_objs
