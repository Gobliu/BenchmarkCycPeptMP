import numpy as np
import pandas as pd
from sklearn.metrics import *
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


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


def multi_scatter(csv_d):
    plt.figure()
    for csv in csv_d:
        split, mode, seed = csv['split'], csv['mode'], csv['seed']
        csv_path = f'../CSV/Predictions/{split}/{mode}/AttentiveFP_seed{seed}.csv'
        print(f'Dealing {csv_path}')
        df = pd.read_csv(csv_path)

        if mode == 'regression':
            true = df.Normalized_PAMPA
            # fpr, tpr, roc_auc = classification_matrices(true / 2 + 0.5, df[f'Pred_{seed}'] / 2 + 0.5)
        elif mode == 'classification' or mode == 'soft':
            true = df.Soft_Label
        plt.scatter(true, df[f'Pred_{seed}'], label=f'{csv_path}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('True')
    plt.ylabel('Predictions')
    plt.legend(loc='lower right')
    plt.show()


def multi_hist(csv_d):
    objs = []
    for csv in csv_d:
        split, mode, seed = csv['split'], csv['mode'], csv['seed']
        csv_path = f'../CSV/Predictions/{split}/{mode}/DMPNN_seed{seed}.csv'
        print(f'Dealing {csv_path}')
        df = pd.read_csv(csv_path)
        objs.append({'value': df[f'Pred_{seed}'], 'source': f'{split}/{mode}/{seed}'})
    df = pd.concat(axis=0, ignore_index=True, objs=[pd.DataFrame.from_dict(i) for i in objs])

    fig = sns.histplot(data=df, x='value', hue='source', bins=10, multiple='dodge', shrink=0.8)
    fig.set_xticks(np.linspace(0, 1, 11, endpoint=True))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.set_xlabel('Predictions Value', fontsize=14)
    fig.set_ylabel('Counts', fontsize=14)
    sns.move_legend(fig, loc='upper right', frameon=False)
    # plt.legend(loc='upper right', fontsize=12)
    plt.show()


if __name__ == '__main__':
    seed_list_ = list(range(1, 11))
    csv_dict = [
        {'split': 'random', 'mode': 'soft', 'seed': 1},
        {'split': 'random', 'mode': 'classification', 'seed': 1},
        ]
    multi_auc(csv_dict)
    # multi_scatter(csv_dict)
    # multi_hist(csv_dict)
