import hashlib
import os


def find_hash_of_file(file_path: str) -> str:
    """
    Find the hash of given image file with its properties
        finding the hash of image content is costly operation
    Args:
        file_path (str): exact image path
    Returns:
        hash (str): digest with sha1 algorithm
    """
    file_stats = os.stat(file_path)

    # some properties
    file_size = file_stats.st_size
    creation_time = file_stats.st_ctime
    modification_time = file_stats.st_mtime

    properties = f"{file_size}-{creation_time}-{modification_time}"

    hasher = hashlib.sha1()
    hasher.update(properties.encode("utf-8"))
    return hasher.hexdigest()
