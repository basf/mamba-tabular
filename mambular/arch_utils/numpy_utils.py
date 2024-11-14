import torch
import numpy as np


def check_numpy(x):
    """Makes sure x is a numpy array"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x
