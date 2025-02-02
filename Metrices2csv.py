import os

import numpy as np
import pandas as pd
from sklearn.metrics import *
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from Metrices import regression_matrices, classification_matrices
# from Metrices import ensemble_pred_regression, ensemble_pred_classification

# def regression_matrices(true, pred):
#     print('Converting to original value')
#     print(f'ground truth: max {true.max():.3f}, min {true.min():.3f}')
#     print(f'prediction: mean {pred.max():.3f}, min {pred.min():.3f}')
#     mae = mean_absolute_error(true, pred)
#     rmse = mean_squared_error(true, pred, squared=False)
#     r2 = r2_score(true, pred)
#     pearson_r, _ = pearsonr(true, pred)
#     print(f"mae {mae:.3f}, rmse {rmse:.3f}, r2 {r2:.3f}, pearson_r {pearson_r:.3f}")
#     return mae, rmse, r2, pearson_r
#
#
# def classification_matrices(true, pred, cutoff=0.5):
#     print('Converting to binary')
#     print(f'ground truth: max {true.max():.3f}, min {true.min():.3f}')
#     print(f'prediction: max {pred.max():.3f}, min {pred.min():.3f}')
#     true[true < 0.5] = 0
#     true[true >= 0.5] = 1
#     true = true.astype(int)
#     fpr, tpr, thresholds = roc_curve(true, pred)
#
#     # Calculate the AUC
#     roc_auc = auc(fpr, tpr)
#
#     pred = pred.copy()
#     pred[pred < cutoff] = 0
#     pred[pred >= cutoff] = 1
#
#     f1 = f1_score(true, pred)
#     print(f"auc: {roc_auc}, f1: {f1:.2f}")
#     acc = accuracy_score(true, pred)
#     precision = precision_score(true, pred)
#     recall = recall_score(true, pred)
#
#     conf_matrix = confusion_matrix(true, pred)
#     print(np.sum(true), np.sum(pred))
#     print("Confusion Matrix:")
#     print(conf_matrix)
#     print('fpr', conf_matrix[0, 1] / np.sum(conf_matrix[0, :]))
#     print('tpr', conf_matrix[1, 1] / np.sum(conf_matrix[1, :]))
#     return roc_auc, f1, acc, precision, recall


def ensemble_pred_regression(csv_files, metric_csv_path):
    mae_list = []
    rmse_list = []
    r2_list = []
    pearson_r_list = []
    auc_list = []

    if os.path.exists(metric_csv_path):
        metric_df = pd.read_csv(metric_csv_path)
    else:
        metric_df = pd.DataFrame(columns=['split', 'mode', 'model', 'seed', 'mae', 'rmse', 'r2', 'pearson_r', 'auc'])

    for csv in csv_files:
        df = pd.read_csv(csv)
        true = df.Normalized_PAMPA

        # Find prediction columns
        # seed_columns = [col for col in df.columns if 'pred' in col.lower()]
        seed_columns = [col for col in df.columns if 'pred_normalized_pampa' in col.lower()]
        print('Column names of predictions:', seed_columns, 'Number of samples:', len(df))

        for col in seed_columns:
            try:
                seed = int(col.split('_')[-1])
            except ValueError:
                seed = int(csv[:-4].split('_')[-1][4:])
            print('column name:', col, 'seed:', seed)

            # Calculate regression and classification metrics
            mae, rmse, r2, pearson_r = regression_matrices(true * 2 - 6, df[col] * 2 - 6)
            auc_score, f1, acc, precision, recall = classification_matrices(true / 2 + 0.5, df[col] / 2 + 0.5)

            # Add the calculated metrics to a new row
            row = {'split': split,
                   'mode': mode,
                   'model': model,
                   'seed': seed,
                   'mae': mae,
                   'rmse': rmse,
                   'r2': r2,
                   'pearson_r': pearson_r,
                   'auc': auc_score}
            metric_df = pd.concat([metric_df, pd.DataFrame([row])], ignore_index=True)

            # Append to metrics lists
            mae_list.append(mae)
            rmse_list.append(rmse)
            r2_list.append(r2)
            pearson_r_list.append(pearson_r)
            auc_list.append(auc_score)

    # Print summary statistics for metrics
    print('~~~~~~~~ metric statistics ~~~~~~~')
    print('split:', split, 'mode:', mode, 'model:', model)
    print('mae:', np.mean(mae_list), np.std(mae_list, ddof=1), mae_list)
    print('rmse:', np.mean(rmse_list), np.std(rmse_list, ddof=1), rmse_list)
    print('r2:', np.mean(r2_list), np.std(r2_list, ddof=1), r2_list)
    print('pearson_r:', np.mean(pearson_r_list), np.std(pearson_r_list, ddof=1), pearson_r_list)
    print('auc:', np.mean(auc_list), np.std(auc_list, ddof=1), auc_list)

    # Save the updated metrics to the CSV file
    metric_df.to_csv(metric_csv_path, index=False)


def ensemble_pred_classification(csv_files, metric_csv_path):
    auc_list = []
    f1_list = []
    acc_list = []
    precision_list = []
    recall_list = []

    if os.path.exists(metric_csv_path):
        metric_df = pd.read_csv(metric_csv_path)
    else:
        metric_df = pd.DataFrame(columns=['split', 'mode', 'model', 'seed', 'acc', 'pre', 'recall', 'f1', 'auc'])

    for csv in csv_files:
        df = pd.read_csv(csv)
        true = np.copy(df.Soft_Label)

        # Calculate the average for each row in the selected columns
        seed_columns = [col for col in df.columns if 'pred' in col.lower()]
        print('Column names of predictions', seed_columns, 'Number of samples', len(df))

        for col in seed_columns:
            # seed = int(col.split('_')[-1])
            seed = csv.split('_')[-1][:-4]
            print('column name:', col, 'seed:', seed)

            auc_score, f1, acc, precision, recall = classification_matrices(true, df[col])
            print(acc, precision)

            # Add the calculated metrics to a new row
            row = {'split': split,
                   'mode': mode,
                   'model': model,
                   'seed': seed,
                   'acc': acc,
                   'pre': precision,
                   'recall': recall,
                   'f1': f1,
                   'auc': auc_score}
            metric_df = pd.concat([metric_df, pd.DataFrame([row])], ignore_index=True)

            auc_list.append(auc_score)
            f1_list.append(f1)
            acc_list.append(acc)
            precision_list.append(precision)
            recall_list.append(recall)

    print('~~~~~~~~ metric statistics ~~~~~~~')
    print('acc', np.mean(acc_list), np.std(acc_list, ddof=1), acc_list)
    print('precision', np.mean(precision_list), np.std(precision_list, ddof=1), precision_list)
    print('recall', np.mean(recall_list), np.std(recall_list, ddof=1), recall_list)
    print('f1', np.mean(f1_list), np.std(f1_list, ddof=1), f1_list)
    print('auc', np.mean(auc_list), np.std(auc_list, ddof=1), acc_list)
    metric_df.to_csv(metric_csv_path, index=False)


def combine_csv(csv_list):
    dfs = [pd.read_csv(file) for file in csv_list]
    dfs = pd.concat(dfs, ignore_index=True)
    dfs.to_csv('temp.csv', index=False)
    if mode == 'regression':
        ensemble_pred_regression(['temp.csv'], metric_csv)
    elif mode == 'classification' or mode == 'soft':
        ensemble_pred_classification(['temp.csv'], metric_csv)


if __name__ == '__main__':
    seed_list_ = list(range(1, 11))
    split = 'random'
    mode = 'soft'
    model = 'RNN'
    metric_csv = f'./CSV/Predictions/metric_{mode}.csv'

    # csv_file = [f'./CSV/Predictions/{split}/{mode}/{model}_seed{i}.csv' for i in seed_list_]
    # # csv_file = [f'./CSV/Predictions/{split}/{mode}/{model}.csv']

    # if mode == 'regression':
    #     ensemble_pred_regression(csv_file, metric_csv)
    # elif mode == 'classification' or mode == 'soft':
    #     ensemble_pred_classification(csv_file, metric_csv)

    # csv_file = [f'./PytorchModels/test_1.csv']
    csv_file = [f'./CSV/Predictions/{split}/{mode}/{model}_ipsize128_hsize128_numlayer2_lr0.0001_{i}.csv' for i in seed_list_]
    # ensemble_pred_regression(csv_file, metric_csv)
    ensemble_pred_classification(csv_file, metric_csv)
    # print(seed_list_)
    # print(csv_file)
    # csv_file = [f'./CSV/Predictions/{split}/{mode}/{model}_mol_length_8.csv',
    #             f'./CSV/Predictions/{split}/{mode}/{model}_mol_length_9.csv']
    # csv_file = [f'./CSV/Predictions/{split}/{mode}/{model}_mol_length89.csv']
    # combine_csv(csv_file)

    # print(true_pm)
    # print(pred_pm)
    # regression_matrices(true_pm, pred_pm)
    # classification_matrices(true_pm, pred_pm)
