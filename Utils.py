import random
import numpy as np


def set_seed(seed: int, tensorflow: bool = True, pytorch: bool = True):
    """
    Set the random seed for reproducibility across NumPy, TensorFlow, PyTorch, and Python's random module.

    Parameters:
    - seed (int): The random seed value.
    - tensorflow (bool, optional): If True, set the seed for TensorFlow. Defaults to True.
    - pytorch (bool, optional): If True, set the seed for PyTorch. Defaults to True.

    Raises:
    - ImportError: If TensorFlow or PyTorch is requested but not installed.
    """

    # Set seed for Python's random module and NumPy
    random.seed(seed)
    np.random.seed(seed)

    # Set seed for TensorFlow
    if tensorflow:
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            print("Warning: TensorFlow is not installed. Skipping TensorFlow seed setting.")

    # Set seed for PyTorch
    if pytorch:
        try:
            import torch
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            print("Warning: PyTorch is not installed. Skipping PyTorch seed setting.")
