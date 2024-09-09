from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_absolute_error

from DataLoader import loader_random_split_scaled, loader_scaffold_split_scaled


def random_regression():
    for split_seed in range(1, 11):
        data = loader_random_split_scaled(split_seed)
        test_df = data['test_df']
        # print(data['train'][0])
        svr = SVR().fit(data['train'][0], data['train'][1])
        y_pred = svr.predict(data['test'][0])
        print(mean_absolute_error(data['test'][1], y_pred))
        test_df[f'True_{split_seed}'] = data['test'][1]
        test_df[f'Pred_{split_seed}'] = y_pred
        test_csv_path = f"../CSV/Predictions/random/regression/SVM_seed{split_seed}.csv"
        print('Saving csv of test data to', test_csv_path)
        test_df.to_csv(test_csv_path, index=False)


def scaffold_regression():
    mol_length_list = [6, 7, 10]
    train_list = [f'../CSV/Data/mol_length_{i}_train.csv' for i in mol_length_list]
    test_list = [f'../CSV/Data/mol_length_{i}_test.csv' for i in mol_length_list]
    data = loader_scaffold_split_scaled(train_list, test_list)
    test_df = data['test_df']
    print(data['train'][0])
    svr = SVR().fit(data['train'][0], data['train'][1])
    y_pred = svr.predict(data['test'][0])
    print(mean_absolute_error(data['test'][1], y_pred))
    test_df[f'True'] = data['test'][1]
    test_df[f'Pred'] = y_pred
    test_csv_path = f"../CSV/Predictions/random/regression/SVM.csv"
    print('Saving csv of test data to', test_csv_path)
    test_df.to_csv(test_csv_path, index=False)


if __name__ == '__main__':
    # split_list = ['random', 'scaffold']
    # task_list = ['regression', 'classification']
    # for split_ in split_list:
    #     for task_ in task_list:
    #         main(split_, task_)
    # random_regression()
    scaffold_regression()
