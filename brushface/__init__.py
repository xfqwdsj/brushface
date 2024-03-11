try:
    import keras.config

    if keras.config.backend() != "torch":
        raise ModuleNotFoundError()
except ModuleNotFoundError:
    import warnings

    warnings.warn(
        "Keras is not using PyTorch backend. This may cause unexpected behavior. "
        "You can set the backend to PyTorch by setting the KERAS_BACKEND environment variable to `torch`."
    )

from brushface.modules.analysis import analyze, analyze_from_faces
from brushface.modules.cli import cli
from brushface.modules.datastore import (
    get_datastore_path,
    load_datastore,
    load_datastore_or_empty,
    load_datastore_or_none,
    save_datastore,
    update_datastore,
)
from brushface.modules.detection import extract_faces
from brushface.modules.recognition import find, find_from_faces
from brushface.modules.representation import represent, represent_from_faces
from brushface.modules.verification import verify

__all__ = [
    "cli",
    "analyze",
    "analyze_from_faces",
    "extract_faces",
    "find",
    "find_from_faces",
    "get_datastore_path",
    "load_datastore",
    "load_datastore_or_empty",
    "load_datastore_or_none",
    "save_datastore",
    "update_datastore",
    "represent",
    "represent_from_faces",
    "verify",
]

__version__ = "0.1.0"
