from pathlib import Path
from typing import Type, TypeVar, Union

import numpy as np

from brushface.models.abstract import Detector, Recognizer

type _Path = Union[str, Path]
type _Img = Union[_Path, np.ndarray]
type _Detector = Union[Type[Detector], Detector]
type _Recognizer = Union[Type[Recognizer], Recognizer]


def get_path(target: _Path) -> Path:
    if isinstance(target, Path):
        return target
    return Path(target)


EXTRACT_TYPE_T = TypeVar("EXTRACT_TYPE_T")


def extract_type(target: Union[Type[EXTRACT_TYPE_T], EXTRACT_TYPE_T]) -> EXTRACT_TYPE_T:
    if isinstance(target, type):
        return target()
    return target
