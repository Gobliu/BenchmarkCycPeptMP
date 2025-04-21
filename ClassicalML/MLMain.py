import os
import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, f1_score

# Dynamically append project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Utils import set_seed, get_combined_args, get_csv_list
from ClassicalML.DataLoader import loader_random_split_scaled, loader_scaffold_split_scaled


def initialize_model(mode, task):
    """
    Initializes and returns the specified ML model.

    Parameters:
    - mode (str): 'SVM' or 'RF'
    - task (str): 'regression' or 'classification'

    Returns:
    - model (sklearn model): Initialized machine learning model.

    Raises:
    - ValueError: If mode is not 'SVM' or 'RF'.
    - ValueError: If task is not 'regression' or 'classification'.
    """
    if mode not in ['SVM', 'RF']:
        raise ValueError(f"Invalid mode: '{mode}'. Expected 'SVM' or 'RF'.")

    if task not in ['regression', 'classification']:
        raise ValueError(f"Invalid task: '{task}'. Expected 'regression' or 'classification'.")

    if task == 'regression':
        return SVR() if mode == 'SVM' else RandomForestRegressor()
    else:  # classification
        return SVC(probability=True) if mode == 'SVM' else RandomForestClassifier()


def preprocess_labels(y):
    """
    Converts permeability values into binary labels for classification.

    Parameters:
    - y (pd.Series): Target variable.

    Returns:
    - y (pd.Series): Binary classification labels.
    """
    y = y.copy()  # Avoid modifying original data
    y[y >= -6] = 1
    y[y < -6] = 0
    return y.astype(np.uint8)


def train_model(model, x_train, y_train, is_classification=False):
    """
    Trains the given model on the training data.

    Parameters:
    - model: Machine learning model.
    - x_train (array): Training features.
    - y_train (array): Training labels.
    - is_classification (bool): Whether it's a classification task.

    Returns:
    - Trained model.
    """
    if is_classification:
        y_train = preprocess_labels(y_train)

    model.fit(x_train, y_train)
    return model


def evaluate_model(model, data, split_seed, test_csv_path, is_classification=False):
    """
    Evaluates the trained model on test data and saves results.

    Parameters:
    - model: Trained machine learning model.
    - data (dict): Dictionary containing training and test data.
    - split_seed (int): The seed used for splitting.
    - test_csv_path (str): Path to save predictions.
    - is_classification (bool): Whether it's a classification task.
    """
    x_test, y_test = data['test']

    if is_classification:
        y_test = preprocess_labels(y_test)
        pred_test = model.predict_proba(x_test)[:, 1]
        metric = f1_score(y_test, pred_test.round())
        print(f"F1 Score (Seed {split_seed}):", metric)
    else:
        pred_test = model.predict(x_test)
        metric = mean_absolute_error(y_test, pred_test)
        print(f"MAE (Seed {split_seed}):", metric)

    # Normalize and save results
    test_df = data['test_df']
    test_df['Soft_Label' if is_classification else 'Normalized_PAMPA'] = (y_test + 6) / 2
    test_df[f'Pred_{split_seed}'] = (pred_test + 6) / 2 if not is_classification else pred_test

    print(f"Saving CSV of test data to: {test_csv_path}")
    test_df.to_csv(test_csv_path, index=False)


def main():
    """
    Runs regression or classification experiments with random or scaffold splits.

    Parameters:
    - mode (str): 'SVM' or 'RF'
    - task (str): 'regression' or 'classification'
    - split_type (str): 'random' or 'scaffold'
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    args = get_combined_args()
    assert args.model in ['SVM', 'RF'], f'MLMain.py does not handle model {args.model}...'
    assert args.mode != 'soft', f'!!! SVM and RF has no classification with soft label'

    is_classification = (args.mode == 'classification')
    print(f"Running model: {args.model} mode: {args.mode} split: {args.split}")

    for split_seed in range(1, args.repeat+1):
        set_seed(123 * split_seed ** 2)

        if args.split == 'random':
            data = loader_random_split_scaled(split_seed)
        else:  # scaffold split
            csv_list = get_csv_list(args)
            data = loader_scaffold_split_scaled(csv_list[0], csv_list[2])

        test_csv_path = os.path.join(cur_dir,
                                     f"../CSV/Predictions/{args.split}/{args.mode}/{args.model}_seed{split_seed}.csv")

        # Initialize and train model
        model = initialize_model(args.model, args.mode)
        model = train_model(model, data['train'][0], data['train'][1], is_classification)

        # ==== Uncomment below to run inference on mol length 8 and 9 only
        # test_list = [f'../CSV/Data/mol_length_{i}.csv' for i in [8, 9]]
        # data = loader_scaffold_split_scaled(test_list, test_list)
        # test_csv_path = f"../CSV/Predictions/scaffold/{args.mode}/{args.model}_mol_length89_seed{split_seed}.csv"
        # ====

        # Evaluate model
        evaluate_model(model, data, split_seed, test_csv_path, is_classification)


if __name__ == '__main__':
    main()
