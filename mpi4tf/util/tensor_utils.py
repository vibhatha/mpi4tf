import tensorflow as tf
import numpy as np


def to_tensor(ndarray: np.ndarray):
    tensor = tf.convert_to_tensor(ndarray, dtype=ndarray.dtype)
    return tensor
