import numpy as np
import pandas as pd
from sklearn.metrics import *
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def regression_matrices(true, pred):
    print('Converting to original value')
    print(f'ground truth: max {true.max():.3f}, min {true.min():.3f}')
    print(f'prediction: mean {pred.max():.3f}, min {pred.min():.3f}')
    mae = mean_absolute_error(true, pred)
    rmse = mean_squared_error(true, pred, squared=False)
    r2 = r2_score(true, pred)
    pearson_r, _ = pearsonr(true, pred)
    print(f"mae {mae:.3f}, rmse {rmse:.3f}, r2 {r2:.3f}, pearson_r {pearson_r:.3f}")
    return mae, rmse, r2, pearson_r


def classification_matrices(true, pred, cutoff=0.5):
    print('Converting to binary')
    print(f'ground truth: max {true.max():.3f}, min {true.min():.3f}')
    print(f'prediction: max {pred.max():.3f}, min {pred.min():.3f}')
    true[true < 0.5] = 0
    true[true >= 0.5] = 1
    true = true.astype(int)
    fpr, tpr, thresholds = roc_curve(true, pred)

    # Plot the ROC curve
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc='lower right')
    # plt.show()

    # Calculate the AUC
    roc_auc = auc(fpr, tpr)

    # for t, p in zip(true, pred):
    #     print(t, p)
    pred = pred.copy()
    pred[pred < cutoff] = 0
    pred[pred >= cutoff] = 1

    f1 = f1_score(true, pred)
    print(f"auc: {roc_auc}, f1: {f1:.2f}")
    acc = accuracy_score(true, pred)
    precision = precision_score(true, pred)
    recall = recall_score(true, pred)

    conf_matrix = confusion_matrix(true, pred)
    print(np.sum(true), np.sum(pred))
    print("Confusion Matrix:")
    print(conf_matrix)
    print('fpr', conf_matrix[0, 1] / np.sum(conf_matrix[0, :]))
    print('tpr', conf_matrix[1, 1] / np.sum(conf_matrix[1, :]))
    # print(np.sum(true > 0.5))
    # print(acc, precision)
    return roc_auc, f1, acc, precision, recall


def ensemble_pred_regression(csv_files):
    mae_list = []
    rmse_list = []
    r2_list = []
    pearson_r_list = []
    auc_list = []
    for csv in csv_files:
        df = pd.read_csv(csv)
        true = df.Normalized_PAMPA
        # print(true * 2 - 6)

        # Calculate the average for each row in the selected columns
        seed_columns = [col for col in df.columns if 'pred' in col.lower()]
        print('Column names of predictions', seed_columns, 'Number of samples', len(df))

        for col in seed_columns:
            print('column name', col)
            # assert min(df[col]) > -13, f'{min(df[col])}'
            # mae, rmse, r2, pearson_r = regression_matrices(true, df[col])
            mae, rmse, r2, pearson_r = regression_matrices(true * 2 - 6, df[col] * 2 - 6)
            auc_score, f1, acc, precision, recall = classification_matrices(true / 2 + 0.5, df[col] / 2 + 0.5)
            print(acc, precision)

            mae_list.append(mae)
            rmse_list.append(rmse)
            r2_list.append(r2)
            pearson_r_list.append(pearson_r)

            auc_list.append(auc_score)

    print('~~~~~~~~ metric statistics ~~~~~~~')
    # print(mae_list)
    print('mae', np.mean(mae_list), np.std(mae_list))
    print('rmse', np.mean(rmse_list), np.std(rmse_list))
    # print(r2_list)
    print('r2', np.mean(r2_list), np.std(r2_list))
    print('pearson_r', np.mean(pearson_r_list), np.std(pearson_r_list))
    print('auc', np.mean(auc_list), np.std(auc_list))


def ensemble_pred_classification(csv_files):
    auc_list = []
    f1_list = []
    acc_list = []
    precision_list = []
    recall_list = []
    for csv in csv_files:
        df = pd.read_csv(csv)
        true = np.copy(df.Soft_Label)

        # Calculate the average for each row in the selected columns
        seed_columns = [col for col in df.columns if 'pred' in col.lower()]
        print('Column names of predictions', seed_columns, 'Number of samples', len(df))

        for col in seed_columns:
            print('column name', col)

            auc_score, f1, acc, precision, recall = classification_matrices(true, df[col], cutoff=0.16)
            print(acc, precision)

            auc_list.append(auc_score)
            f1_list.append(f1)
            acc_list.append(acc)
            precision_list.append(precision)
            recall_list.append(recall)

    print('~~~~~~~~ metric statistics ~~~~~~~')
    print('acc', np.mean(acc_list), np.std(acc_list))
    print('precision', np.mean(precision_list), np.std(precision_list))
    # print(len(recall_list), recall_list)
    print('recall', np.mean(recall_list), np.std(recall_list))
    print('f1', np.mean(f1_list), np.std(f1_list))
    print('auc', np.mean(auc_list), np.std(auc_list))
    print(auc_list)


if __name__ == '__main__':
    seed_list_ = list(range(1, 11))
    split = 'scaffold'
    mode = 'classification'
    model = 'GCN'
    csv_file = [f'./CSV/Predictions/{split}/{mode}/{model}_seed{i}.csv' for i in seed_list_]
    # csv_file = ['./CSV/Predictions/TVT_Scaffold_Split/Trained_on_6&7&10/Regression/DMPNN.csv']
    # csv_file = [f'./CSV/Predictions/{split}/{mode}/{model}_mol_length_8.csv',
    #             f'./CSV/Predictions/{split}/{mode}/{model}_mol_length_9.csv']
    if mode == 'regression':
        ensemble_pred_regression(csv_file)
    elif mode == 'classification' or mode == 'soft':
        ensemble_pred_classification(csv_file)
    # print(true_pm)
    # print(pred_pm)
    # regression_matrices(true_pm, pred_pm)
    # classification_matrices(true_pm, pred_pm)

