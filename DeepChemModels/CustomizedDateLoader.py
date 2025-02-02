import pandas as pd
import deepchem as dc
import numpy as np


def data_loader_separate(csv_list, loader):
    train = loader.create_dataset(csv_list[0])
    valid = loader.create_dataset(csv_list[1])
    test = loader.create_dataset(csv_list[2])
    print(f'Got {len(train.y)} samples for train, {len(valid.y)} for valid, and {len(test.y)} for test')
    df_list = []
    for f in csv_list[2]:
        df_list.append(pd.read_csv(f))
    df = pd.concat(df_list, ignore_index=True)
    return {'train': train, 'valid': valid, 'test': test, 'test_df': df}


def data_loader_all_in_one(csv_list, split_seed, loader):
    df_list = []
    for f in csv_list:
        print(f)
        df_list.append(pd.read_csv(f))
    ip_df = pd.concat(df_list, ignore_index=True)
    grouped = ip_df.groupby(f'split{split_seed}')
    for group_name, group_df in grouped:
        group_df.to_csv(f'temp_{group_name}.csv', index=False)
    train = loader.create_dataset('temp_train.csv')
    valid = loader.create_dataset('temp_valid.csv')
    test = loader.create_dataset('temp_test.csv')
    print(f'Got {len(train.y)} samples for train, {len(valid.y)} for valid, and {len(test.y)} for test')
    df = pd.read_csv('temp_test.csv')
    return {'train': train, 'valid': valid, 'test': test, 'test_df': df}


def convert_multitask(dataset, task_number):
    """Converts a single-task dataset to a multitask dataset by repeating labels and weights.

    Args:
        dataset (dc.data.Dataset): The original single-task dataset.
        task_number (int): The number of tasks in the converted multitask dataset.

    Returns:
        dc.data.NumpyDataset: The converted multitask dataset.

    Raises:
        ValueError: If the shapes of labels or weights are not compatible with repetition.
    """

    # Assuming features, labels, and weights
    new_dataset = dc.data.NumpyDataset(
        X=dataset.X,
        y=np.repeat(dataset.y, task_number, axis=1),
        w=np.repeat(dataset.w, task_number, axis=1),
        ids=dataset.ids
    )
    return new_dataset


def adjust_task_weight(dataset, weights):
    # Assuming features, labels, and weights
    new_dataset = dc.data.NumpyDataset(
        X=dataset.X,
        y=dataset.y,
        w=np.repeat(np.asarray(weights)[None], len(dataset.X), axis=0),
        ids=dataset.ids
    )
    return new_dataset


def p2distribution(dataset):
    print("Running p2distribution")
    y = np.copy(dataset.y)
    print("Before y shape", y.shape)
    # y = y[:, None]
    y = np.concatenate((1-y, y), axis=-1)
    print("After y shape", y.shape)
    new_dataset = dc.data.NumpyDataset(
        X=dataset.X,
        y=y,
        w=dataset.w,
        ids=dataset.ids
    )
    return new_dataset


def adjust_class_weights(train_data, valid_data, test_data, weight_list):
    assert train_data.w.shape[-1] == len(weight_list), 'Number of task does match with weight list length...'
    train_w = np.copy(train_data.w)
    valid_w = np.copy(valid_data.w)
    test_w = np.copy(test_data.w)
    for i in range(len(weight_list)):
        print(f'Working on label {i}')
        total = train_data.y.shape[0]
        positive = np.sum(train_data.y[:, i])
        negative = total - positive
        print(f'In total {total} samples in train data, {positive} positive, {negative} negative')
        pw = total * weight_list[i] / (positive * 2)
        nw = total * weight_list[i] / (negative * 2)

        print(train_data.w[:5, i], valid_data.w[:5, i], test_data.w[:5, i])
        print(np.sum(train_data.w), np.sum(valid_data.w), np.sum(test_data.w))
        print(f'Total weight of train data before reweight:', np.sum(train_w[:, i]))
        p_condition = train_data.y[:, i] >= 0.5
        train_w[p_condition] = pw
        n_condition = train_data.y[:, i] < 0.5     # cannot use ~p_condition
        train_w[n_condition] = nw
        print(f'Total weight of train data after reweight:', np.sum(train_w[:, i]))

        print(f'Total weight of valid data before reweight:', np.sum(valid_w[:, i]))
        p_condition = valid_data.y[:, i] >= 0.5
        valid_w[p_condition] = pw
        n_condition = valid_data.y[:, i] < 0.5     # cannot use ~p_condition
        valid_w[n_condition] = nw
        print(f'Total weight of valid data after reweight:', np.sum(valid_w[:, i]))

        print(f'Total weight of test data before reweight:', np.sum(test_w[:, i]))
        p_condition = test_data.y[:, i] >= 0.5
        test_w[p_condition] = pw
        n_condition = test_data.y[:, i] < 0.5     # cannot use ~p_condition
        test_w[n_condition] = nw
        print(f'Total weight of train data after reweight:', np.sum(test_w[:, i]))

        print(train_w[:5, i], valid_w[:5, i], test_w[:5, i])

        train_data = dc.data.NumpyDataset(X=train_data.X, y=train_data.y, w=train_w, ids=train_data.ids)
        valid_data = dc.data.NumpyDataset(X=valid_data.X, y=valid_data.y, w=valid_w, ids=valid_data.ids)
        test_data = dc.data.NumpyDataset(X=test_data.X, y=test_data.y, w=test_w, ids=test_data.ids)

        total = valid_data.y.shape[0]
        positive = np.sum(valid_data.y[:, i])
        negative = total - positive
        print(f'In total {total} samples in valid data, {positive} positive, {negative} negative')

        total = test_data.y.shape[0]
        positive = np.sum(test_data.y[:, i])
        negative = total - positive
        print(f'In total {total} samples in test data, {positive} positive, {negative} negative')
        return train_data, valid_data, test_data


def soft_label2hard(dataset):
    new_y = np.copy(dataset.y)
    new_y[new_y < 0.5] = 0
    new_y[new_y >= 0.5] = 1
    new_dataset = dc.data.NumpyDataset(
        X=dataset.X,
        y=new_y,
        w=dataset.w,
        ids=dataset.ids
    )
    return new_dataset
