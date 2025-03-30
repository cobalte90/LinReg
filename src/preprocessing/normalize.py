import numpy as np
import pandas as pd
from typing import Iterable

def min_max(array: Iterable[float]):
    min_value = np.min(array)
    max_value = np.max(array)
    normalized_array = (array - min_value) / (max_value - min_value)
    return normalized_array


def z_normal(array: Iterable[float]):
    std = np.std(array)
    mean = np.mean(array)
    normalized_array = (array - mean) / std
    return normalized_array
