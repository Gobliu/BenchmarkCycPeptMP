import sys

import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, f1_score

from DataLoader import loader_random_split_scaled, loader_scaffold_split_scaled

sys.path.append('../')
from Utils import set_seed


def random_regression(mode):
    for split_seed in range(1, 11):
        set_seed(123 * split_seed ** 2)
        data = loader_random_split_scaled(split_seed)
        test_df = data['test_df']

        if mode == 'SVM':
            model = SVR()
        elif mode == 'RF':
            model = RandomForestRegressor()

        model.fit(data['train'][0], data['train'][1])
        pred_test = model.predict(data['test'][0])
        print('mae', mean_absolute_error(data['test'][1], pred_test))
        test_df['Normalized_PAMPA'] = (data['test'][1] + 6) / 2
        test_df[f'Normalized_Pred_{split_seed}'] = (pred_test + 6) / 2
        test_csv_path = f"../CSV/Predictions/random/regression/{mode}_seed{split_seed}.csv"
        print('Saving csv of test data to', test_csv_path)
        test_df.to_csv(test_csv_path, index=False)


def scaffold_regression(mode):
    for split_seed in range(1, 11):
        set_seed(123 * split_seed ** 2)
        mol_length_list = [6, 7, 10]
        train_list = [f'../CSV/Data/mol_length_{i}_train.csv' for i in mol_length_list]
        test_list = [f'../CSV/Data/mol_length_{i}_test.csv' for i in mol_length_list]
        data = loader_scaffold_split_scaled(train_list, test_list)
        test_df = data['test_df']

        if mode == 'SVM':
            model = SVR()
        elif mode == 'RF':
            model = RandomForestRegressor()

        model.fit(data['train'][0], data['train'][1])
        pred_test = model.predict(data['test'][0])
        print('mae', mean_absolute_error(data['test'][1], pred_test))
        test_df['Normalized_PAMPA'] = (data['test'][1] + 6) / 2
        test_df[f'Normalized_Pred_{split_seed}'] = (pred_test + 6) / 2
        test_csv_path = f"../CSV/Predictions/scaffold/regression/{mode}_seed{split_seed}.csv"
        print('Saving csv of test data to', test_csv_path)
        test_df.to_csv(test_csv_path, index=False)


def random_classification(mode):
    for split_seed in range(1, 11):
        set_seed(123 * split_seed ** 2)
        data = loader_random_split_scaled(split_seed)
        test_df = data['test_df']

        y_train = data['train'][1]
        y_train[y_train >= -6] = 1
        y_train[y_train < -6] = 0
        y_train = y_train.astype(np.uint8)

        y_test = data['test'][1]
        y_test[y_test >= -6] = 1
        y_test[y_test < -6] = 0
        y_test = y_test.astype(np.uint8)

        if mode == 'SVM':
            model = SVC(probability=True)
        elif mode == 'RF':
            model = RandomForestClassifier()

        model.fit(data['train'][0], y_train)
        pred_test = model.predict_proba(data['test'][0])[:, 1]
        # print(f1_score(data['test'][1], pred_test))
        test_df['Soft_Label'] = y_test
        test_df[f'Pred_{split_seed}'] = pred_test
        test_csv_path = f"../CSV/Predictions/random/classification/{mode}_seed{split_seed}.csv"
        print('Saving csv of test data to', test_csv_path)
        test_df.to_csv(test_csv_path, index=False)


def scaffold_classification(mode):
    for split_seed in range(1, 11):
        set_seed(123 * split_seed ** 2)
        mol_length_list = [6, 7, 10]
        train_list = [f'../CSV/Data/mol_length_{i}_train.csv' for i in mol_length_list]
        test_list = [f'../CSV/Data/mol_length_{i}_test.csv' for i in mol_length_list]
        data = loader_scaffold_split_scaled(train_list, test_list)
        test_df = data['test_df']

        y_train = data['train'][1]
        y_train[y_train >= -6] = 1
        y_train[y_train < -6] = 0
        y_train = y_train.astype(np.uint8)

        y_test = data['test'][1]
        y_test[y_test >= -6] = 1
        y_test[y_test < -6] = 0
        y_test = y_test.astype(np.uint8)

        if mode == 'SVM':
            model = SVC(probability=True)
        elif mode == 'RF':
            model = RandomForestClassifier()

        model.fit(data['train'][0], y_train)
        # pred_test = model.predict(data['test'][0])
        # print(f1_score(data['test'][1], pred_test))
        # print(pred_test[:10])
        pred_test = model.predict_proba(data['test'][0])[:, 1]
        # print(pred_test[:10])
        # quit()

        test_df['Soft_Label'] = y_test
        test_df[f'Pred_{split_seed}'] = pred_test
        test_csv_path = f"../CSV/Predictions/scaffold/classification/{mode}_seed{split_seed}.csv"
        print('Saving csv of test data to', test_csv_path)
        test_df.to_csv(test_csv_path, index=False)


if __name__ == '__main__':
    # split_list = ['random', 'scaffold']
    # task_list = ['regression', 'classification']
    # for split_ in split_list:
    #     for task_ in task_list:
    #         main(split_, task_)

    mode_ = 'SVM'
    random_regression(mode_)
    scaffold_regression(mode_)

    random_classification(mode_)
    scaffold_classification(mode_)
