import random
import numpy as np
import argparse
import yaml
import os


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


def get_combined_args(config_path='Config.yaml'):
    # Resolve absolute path and base directory
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, config_path)

    # Load YAML configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve relative paths in specific keys
    for key in ['model_dir', 'csv_dir']:
        if key in config and not os.path.isabs(config[key]):
            config[key] = os.path.normpath(os.path.join(base_dir, config[key]))

    # Create parser with defaults from YAML
    parser = argparse.ArgumentParser()
    for key, value in config.items():
        arg_type = type(value) if value is not None else str
        parser.add_argument(f'--{key}', default=value, type=arg_type)

    return parser.parse_args()


def get_csv_list(args):
    """
    Returns a list of CSV paths based on the split strategy.

    Args:
        args: Argument namespace containing 'split' field.

    Returns:
        List or nested list of CSV file paths for training, validation, and test splits.

    Raises:
        ValueError: If `args.split` is not one of ['scaffold', 'random'].
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CSV', 'Data')

    if args.split == 'scaffold':
        mol_lengths = [6, 7, 10]
        return [
            [os.path.join(data_dir, f'mol_length_{i}_train.csv') for i in mol_lengths],
            [os.path.join(data_dir, f'mol_length_{i}_valid.csv') for i in mol_lengths],
            [os.path.join(data_dir, f'mol_length_{i}_test.csv') for i in mol_lengths],
        ]

    elif args.split == 'random':
        return [os.path.join(data_dir, "Random_Split.csv")]

    else:
        raise ValueError(f"Unsupported split type: {args.split}. Must be 'scaffold' or 'random'.")


