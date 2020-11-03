import numpy as np

def to_one_hot(arr, max_val):
    """
    Converts an n-dimensional tensor to an n x max_val one-hot tensor
    """
    out = np.zeros([len(arr), max_val])
    out[np.arange(len(arr)), arr] = 1.
    return out
