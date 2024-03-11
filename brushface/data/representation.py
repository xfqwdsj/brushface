from typing import List, TypedDict

from brushface.data.face_area import FaceArea


class Representation(TypedDict):
    """
    A dictionary containing a face embedding and the rectangular region of a face.

    Attributes:
        embedding: The face embedding vector.
        face_area: The rectangular region of the face in the image.
    """

    embedding: List[float]
    face_area: FaceArea
