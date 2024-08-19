import random
import numpy as np


def set_seed(seed, tensorflow=True, pytorch=True):
    """
    Sets the random seed for various libraries (NumPy, TensorFlow, PyTorch, and Python's random module).

    Parameters:
    - seed (int): The random seed value to set for the libraries.
    - tensorflow (bool, optional): Set the seed for TensorFlow if True. Defaults to True.
    - pytorch (bool, optional): Set the seed for PyTorch if True. Defaults to True.

    Note: Assumes libraries are already imported when setting their respective seeds.
    """

    # Set seed for TensorFlow
    try:
        if tensorflow:
            import tensorflow as tf
            tf.random.set_seed(seed)
    except:
        print("Please import Tensorflow as tf to set its seed.")

    # Set seed for PyTorch
    try:
        if pytorch:
            import torch
            torch.manual_seed(seed)

            # Set seed for PyTorch's CUDA and enforce deterministic behavior
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except:
        print("Please import PyTorch to set its seed.")

    # Set seed for NumPy and Python's random module

    np.random.seed(seed)
    random.seed(seed)
