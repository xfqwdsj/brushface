from typing import Optional, Tuple, TypedDict


class FaceArea(TypedDict):
    """
    A dictionary containing the rectangular region of a face.

    Attributes:
        x: The x-coordinate of the top-left corner of the face rectangle.
        y: The y-coordinate of the top-left corner of the face rectangle.
        w: The width of the face rectangle.
        h: The height of the face rectangle.
        confidence: The confidence score of the face detection.
        left_eye: The x and y coordinates of the left eye.
        right_eye: The x and y coordinates of the right eye.
    """

    x: int
    y: int
    w: int
    h: int
    confidence: float
    left_eye: Optional[Tuple[int, int]]
    right_eye: Optional[Tuple[int, int]]
