import numpy as np
import pandas as pd
from sklearn.metrics import *
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def classification_matrices(true, pred):
    print('Converting to binary')
    print(f'ground truth: max {true.max():.3f}, min {true.min():.3f}')
    print(f'prediction: max {pred.max():.3f}, min {pred.min():.3f}')
    true[true < 0.5] = 0
    true[true >= 0.5] = 1
    true = true.astype(int)
    fpr, tpr, thresholds = roc_curve(true, pred)
    for f, t, tr in zip(fpr, tpr, thresholds):
        print(f, t, tr)
    roc_auc = auc(fpr, tpr)

    pred = pred.copy()
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1

    f1 = f1_score(true, pred)
    print(f"auc: {roc_auc}, f1: {f1:.2f}")
    return fpr, tpr, roc_auc


def multi_auc(csv_d):
    plt.figure()

    for csv in csv_d:
        split, mode, seed = csv['split'], csv['mode'], csv['seed']
        csv_path = f'../CSV/Predictions/{split}/{mode}/AttentiveFP_seed{seed}.csv'
        print(f'Dealing {csv_path}')
        df = pd.read_csv(csv_path)
        true = df.Normalized_PAMPA

        if mode == 'regression':
            fpr, tpr, roc_auc = classification_matrices(true / 2 + 0.5, df[f'Pred_{seed}'] / 2 + 0.5)
        elif mode == 'classification' or mode == 'soft':
            fpr, tpr, roc_auc = classification_matrices(true / 2 + 0.5, df[f'Pred_{seed}'])
        plt.plot(fpr, tpr, lw=2, label=f'{csv_path}\nROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    seed_list_ = list(range(1, 11))
    csv_dict = [
        {'split': 'scaffold', 'mode': 'soft', 'seed': 1},
        {'split': 'scaffold', 'mode': 'classification', 'seed': 1},
        ]
    multi_auc(csv_dict)
