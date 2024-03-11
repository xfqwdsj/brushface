from typing import List, Optional, TypedDict


class RecognitionData(TypedDict):
    img_path: str
    hash: str
    embedding: Optional[List[float]]
    x: int
    y: int
    w: int
    h: int


class RecognitionResult(TypedDict):
    """
    A dictionary containing the result of a face recognition.

    Attributes:
        img_path: The path to the image file.
        hash: The hash of the image file.
        target_x: The x-coordinate of the top-left corner of the face.
        target_y: The y-coordinate of the top-left corner of the face.
        target_w: The width of the face.
        target_h: The height of the face.
        distance: The distance between the source face and the recognized face.
        threshold: The threshold for face recognition.
    """

    img_path: str
    hash: str
    target_x: int
    target_y: int
    target_w: int
    target_h: int
    distance: float
    threshold: float
