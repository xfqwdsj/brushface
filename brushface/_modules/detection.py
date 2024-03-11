from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from keras.preprocessing import image

from brushface._modules import preprocessing
from brushface._modules.type import _Detector, _Img, extract_type
from brushface.data.detected_face import DetectedFace
from brushface.data.face_area import FaceArea
from brushface.internal import Logger
from brushface.models import default_detector
from brushface.models.abstract import Detector

logger = Logger(__name__)


def extract_faces(
    img: _Img,
    target_size: Optional[Tuple[int, int]] = (224, 224),
    grayscale=False,
    detector: Optional[_Detector] = default_detector,
    enforce_detection=True,
    align=True,
    expand_percentage=0,
    return_rgb=True,
) -> List[DetectedFace]:
    """
    Extracts faces from a given image.

    Args:
        img: The path or URL to the image, a numpy array in BGR format, or a base64 encoded image.
        target_size: Final shape of facial image. Black pixels will be added to resize the image.
        detector: The face detector to use.
        enforce_detection: If no face is detected in an image, raise an exception.
            Not enforcing detection can be useful for low-resolution images.
        align: Perform alignment based on the eye positions.
        expand_percentage: The percentage to expand the detected face area.
        grayscale: Convert the image to grayscale before processing.
        return_rgb: Return the image in RGB format instead of 4D BGR format for ML models.

    Returns:
        The results of the face extraction process.
    """

    detector = extract_type(detector)

    resp_objs = []

    # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    img, img_name = preprocessing.load_image(img)

    if img is None:
        raise ValueError(f"Exception while loading {img_name}.")

    base_region: FaceArea = {
        "x": 0,
        "y": 0,
        "w": img.shape[1],
        "h": img.shape[0],
        "confidence": 0,
        "left_eye": None,
        "right_eye": None,
    }

    face_objs: List[DetectedFace] = [{"img": img, "face_area": base_region}]

    if detector is not None:
        detected_faces = detect_faces(
            detector=detector, img=img, align=align, expand_percentage=expand_percentage
        )

        # in case of no face found
        if len(face_objs) == 0:
            if enforce_detection:
                origin = f" in {img_name}" if img_name is not None else ""
                raise ValueError(
                    f"Face could not be detected{origin}. "
                    "Please confirm that the image contains a face or consider to set enforce_detection param to False."
                )
        else:
            face_objs = detected_faces

    for face_obj in face_objs:
        current_img = face_obj["img"]
        current_region = face_obj["face_area"]

        if current_img.shape[0] == 0 or current_img.shape[1] == 0:
            continue

        if grayscale:
            current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

        # resize and padding
        if target_size is not None:
            factor_0 = target_size[0] / current_img.shape[0]
            factor_1 = target_size[1] / current_img.shape[1]
            factor = min(factor_0, factor_1)

            dsize = (
                int(current_img.shape[1] * factor),
                int(current_img.shape[0] * factor),
            )
            current_img = cv2.resize(current_img, dsize)

            diff_0 = target_size[0] - current_img.shape[0]
            diff_1 = target_size[1] - current_img.shape[1]
            if not grayscale:
                # Put the base image in the middle of the padded image
                current_img = np.pad(
                    current_img,
                    (
                        (diff_0 // 2, diff_0 - diff_0 // 2),
                        (diff_1 // 2, diff_1 - diff_1 // 2),
                        (0, 0),
                    ),
                    "constant",
                )
            else:
                current_img = np.pad(
                    current_img,
                    (
                        (diff_0 // 2, diff_0 - diff_0 // 2),
                        (diff_1 // 2, diff_1 - diff_1 // 2),
                    ),
                    "constant",
                )

            # double check: if target image is not still the same size with target.
            if current_img.shape[0:2] != target_size:
                current_img = cv2.resize(current_img, target_size)

        # normalizing the image pixels
        # what this line doing? must?
        img_pixels = image.img_to_array(current_img)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255  # normalize input in [0, 1]
        # discard expanded dimension
        if return_rgb and len(img_pixels.shape) == 4:
            img_pixels = img_pixels[0]

        resp_objs.append(
            {
                "img": img_pixels[:, :, ::-1] if return_rgb else img_pixels,
                "face_area": current_region,
            }
        )

    if len(resp_objs) == 0 and enforce_detection:
        raise ValueError(
            f"Exception while extracting faces from {img_name}."
            "Consider to set enforce_detection arg to False."
        )

    return resp_objs


def detect_faces(
    detector: Detector, img: np.ndarray, align=True, expand_percentage=0
) -> List[DetectedFace]:
    """Detect face(s) from a given image."""

    # validate expand percentage score
    if expand_percentage < 0:
        logger.warn(
            f"Expand percentage cannot be negative but you set it to {expand_percentage}."
            "Overwritten it to 0."
        )
        expand_percentage = 0

    # find facial areas of given image
    facial_areas = detector.detect_faces(img=img)

    results = []
    for facial_area in facial_areas:
        x = facial_area["x"]
        y = facial_area["y"]
        w = facial_area["w"]
        h = facial_area["h"]
        left_eye = facial_area["left_eye"]
        right_eye = facial_area["right_eye"]
        confidence = facial_area["confidence"]

        if expand_percentage > 0:
            # Expand the facial region height and width by the provided percentage
            # ensuring that the expanded region stays within img.shape limits
            expanded_w = w + int(w * expand_percentage / 100)
            expanded_h = h + int(h * expand_percentage / 100)

            x = max(0, x - int((expanded_w - w) / 2))
            y = max(0, y - int((expanded_h - h) / 2))
            w = min(img.shape[1] - x, expanded_w)
            h = min(img.shape[0] - y, expanded_h)

        # extract detected face unaligned
        detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]

        # align original image, then find projection of detected face area after alignment
        if align and left_eye is not None and right_eye is not None:
            aligned_img, angle = align_face(
                img=img, left_eye=left_eye, right_eye=right_eye
            )
            rotated_x1, rotated_y1, rotated_x2, rotated_y2 = rotate_facial_area(
                facial_area=(x, y, x + w, y + h),
                angle=angle,
                size=(img.shape[0], img.shape[1]),
            )
            detected_face = aligned_img[
                int(rotated_y1) : int(rotated_y2), int(rotated_x1) : int(rotated_x2)
            ]

        results.append(
            {
                "img": detected_face,
                "face_area": {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "confidence": confidence,
                    "left_eye": left_eye,
                    "right_eye": right_eye,
                },
            }
        )
    return results


def align_face(
    img: np.ndarray, left_eye: Union[list, tuple], right_eye: Union[list, tuple]
) -> Tuple[np.ndarray, float]:
    """
    Aligns a given image horizontally based on the eye positions.

    Returns:
        The aligned image and the angle of rotation.
    """

    # sometimes unexpectedly detected images come with nil dimensions
    if img.shape[0] == 0 or img.shape[1] == 0:
        return img, 0

    angle = float(
        np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    )
    img = np.array(Image.fromarray(img).rotate(angle))
    return img, angle


def rotate_facial_area(
    facial_area: Tuple[int, int, int, int], angle: float, size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Rotate the facial area around its center.

    Inspired from the work of @UmutDeniz26 - https://github.com/serengil/retinaface/pull/80

    Args:
        facial_area: The (x1, y1, x2, y2) of the facial area.
        angle: Angle of rotation in degrees. Its sign determines the direction of rotation.
            Note that angles > 360 degrees are normalized to the range [0, 360).
        size: Tuple representing the size of the image (width, height).

    Returns:
        The new coordinates (x1, y1, x2, y2) of the rotated facial area.
    """

    # Normalizes the width of the angle, so we don't have to worry about rotations greater than 360 degrees.
    # We take advantage of the weird behavior of the modulation operator for negative values.
    direction = 1 if angle >= 0 else -1
    angle = abs(angle) % 360
    if angle == 0:
        return facial_area

    # Angle in radians
    angle = angle * np.pi / 180

    # Translate the facial area to the center of the image
    x = (facial_area[0] + facial_area[2]) / 2 - size[1] / 2
    y = (facial_area[1] + facial_area[3]) / 2 - size[0] / 2

    # Rotate the facial area
    x_new = x * np.cos(angle) + y * direction * np.sin(angle)
    y_new = -x * direction * np.sin(angle) + y * np.cos(angle)

    # Translate the facial area back to the original position
    x_new = x_new + size[1] / 2
    y_new = y_new + size[0] / 2

    # Calculate the new facial area
    x1 = x_new - (facial_area[2] - facial_area[0]) / 2
    y1 = y_new - (facial_area[3] - facial_area[1]) / 2
    x2 = x_new + (facial_area[2] - facial_area[0]) / 2
    y2 = y_new + (facial_area[3] - facial_area[1]) / 2

    return int(x1), int(y1), int(x2), int(y2)
