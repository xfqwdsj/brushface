import hashlib
from pathlib import Path


def find_hash_of_file(path: Path) -> str:
    """
    Finds the hash of given file with its properties.

    Args:
        path: The path to the file.
    Returns:
        The hash of the file.
    """

    file_stats = path.stat()

    properties = f"{file_stats.st_size}:{file_stats.st_birthtime}:{file_stats.st_ctime}:{file_stats.st_mtime}"

    hasher = hashlib.sha1()
    hasher.update(properties.encode("utf-8"))
    return hasher.hexdigest()
