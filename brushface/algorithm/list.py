from typing import List, Tuple

import numpy as np


def list_to_np_array(*args: List) -> Tuple[np.ndarray, ...]:
    """
    Convert list to NumPy array

    Args:
        *args (List): list of lists

    Returns:
        np.ndarray: tuple of NumPy arrays
    """
    return tuple(np.array(arg) for arg in args)
