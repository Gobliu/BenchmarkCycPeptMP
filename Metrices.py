import numpy as np
import pandas as pd
from sklearn.metrics import *
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def regression_matrices(true, pred):
    print(f'ground truth: mean {true.mean():.3f}, std {true.std():.3f}')

    mae = mean_absolute_error(true, pred)
    rmse = mean_squared_error(true, pred, squared=False)
    r2 = r2_score(true, pred)
    pearson_r, _ = pearsonr(true, pred)
    print(f"mae {mae:.3f}, rmse {rmse:.3f}, r2 {r2:.3f}, pearson_r {pearson_r:.3f}")
    return mae, rmse, r2, pearson_r


def classification_matrices(true, pred):
    true = (true.values + 12) / 12
    pred = (pred.values + 12) / 12
    # print(df.true_pm)
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

    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1

    f1 = f1_score(true, pred)
    print(f"auc: {roc_auc}, f1: {f1:.2f}")

    # conf_matrix = confusion_matrix(true, pred)
    # print("Confusion Matrix:")
    # print(conf_matrix)
    # print(np.sum(true > 0.5))
    return roc_auc, f1


def ensemble_pred(csv_files):
    columns_dict = {}

    # Iterate over each CSV file and group columns by name
    for csv in csv_files:
        current_data = pd.read_csv(csv)

        # Group columns by name in the dictionary
        for column_name in current_data.columns:
            if column_name not in columns_dict:
                columns_dict[column_name] = []

            columns_dict[column_name].append(current_data[column_name])

    # Concatenate columns based on their names
    concatenated_data = {name: pd.concat(columns, axis=0, ignore_index=True) for name, columns in columns_dict.items()}
    df = pd.DataFrame(concatenated_data)
    df.to_csv('./DMPNN_10Blocks.csv', index=False)
    rows_with_nan = df[df.isnull().any(axis=1)]
    print('rows with nan value', rows_with_nan)
    true = df.PAMPA

    # Calculate the average for each row in the selected columns
    seed_columns = [col for col in df.columns if 'pred' in col.lower()]
    print('Column names of predictions', seed_columns)
    mae_list = []
    rmse_list = []
    r2_list = []
    pearson_r_list = []
    auc_list = []
    f1_list = []
    for col in seed_columns:
        print('column name', col)
        # assert min(df[col]) > -13, f'{min(df[col])}'
        mae, rmse, r2, pearson_r = regression_matrices(true, df[col])
        auc_score, f1 = classification_matrices(true, df[col])

        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
        pearson_r_list.append(pearson_r)
        auc_list.append(auc_score)
        f1_list.append(f1)

    print('~~~~~~~~ metric statistics ~~~~~~~')
    print(mae_list)
    print('mae', np.mean(mae_list), np.std(mae_list))
    print('rmse', np.mean(rmse_list), np.std(rmse_list))
    print('r2', np.mean(r2_list), np.std(r2_list))
    print('pearson_r', np.mean(pearson_r_list), np.std(pearson_r_list))
    print('auc', np.mean(auc_list), np.std(auc_list))
    print('f1', np.mean(f1_list), np.std(f1_list))

    mean_pred = df[seed_columns].mean(axis=1)
    # print(mean_pred)
    mae, rmse, r2, pearson_r = regression_matrices(true, mean_pred)
    print(mae, rmse, r2, pearson_r)


if __name__ == '__main__':
    # csv_file = [f'GCN_block{i}.csv' for i in range(10)]
    csv_file = [f'./CSV/DMPNN_block{i}.csv' for i in range(10)]
    # csv_file = ['3seeds_F64_G2_M2_BDTrue_R512-1024-32_E3_end2end.csv']
    # csv_file = ['./CSV/TVT_Split/Trained_on_6&7&10/DMPNN.csv']
    ensemble_pred(csv_file)
    # print(true_pm)
    # print(pred_pm)
    # regression_matrices(true_pm, pred_pm)
    # classification_matrices(true_pm, pred_pm)
