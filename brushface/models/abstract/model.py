from abc import ABC
from typing import Optional


class BrushFaceModel(ABC):
    """
    BrushFace model base class.

    Notes:
        If you want to use a custom model when processing stream images, please note that the model should be
        initialized outside the loop to avoid the model being reloaded every time the loop runs.
    """

    name: str
    _default_instance: Optional["BrushFaceModel"] = None

    @classmethod
    def default(cls) -> "BrushFaceModel":
        """Creates a new instance of the class, storing it as a singleton if no arguments are provided."""

        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    def __str__(self):
        return self.name
