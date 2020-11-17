import numpy as np

def to_one_hot(arr, max_val, default_val=-1):
    """
    Converts an n-dimensional tensor to an n x max_val one-hot tensor
    """
    out = np.ones([len(arr), max_val]) * default_val
    out[np.arange(len(arr)), arr] = 1.
    return out

def argsort2d(arr):
    """
    Argsorts a 2d array
    sol'n from stackoverflow
    """
    arr1d = arr.ravel()
    _1d_idxs = np.argsort(arr1d)
    _2d_idxs = np.unravel_index(_1d_idxs, arr.shape)
    return np.stack(_2d_idxs, axis=1)
