import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(df, drop_cols_range, target_col='Permeability'):
    """
    Prepares feature and target datasets.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - drop_cols_range (slice or list): Column range to drop.
    - target_col (str): Name of the target column.

    Returns:
    - x (pd.DataFrame): Feature set.
    - y (pd.Series): Clipped target values.
    """
    x = df.drop(df.columns[drop_cols_range], axis=1)
    y = df[target_col].clip(lower=-8, upper=-4)
    return x, y


def scale_features(x_train, x_test):
    """Scales features using StandardScaler."""
    scaler = StandardScaler()
    return scaler.fit_transform(x_train), scaler.transform(x_test)


def loader_random_split_scaled(split_seed):
    """
    Loads and processes random split datasets with scaling.

    Parameters:
    - split_seed (int): Identifier for data split.

    Returns:
    - dict: Processed train and test datasets with DataFrame.
    """
    df_all = pd.read_csv("../CSV/Data/CycPeptMPDB_Peptide_All.csv", low_memory=False)
    df_random_split = pd.read_csv("../CSV/Data/Random_Split.csv")

    grouped = df_random_split.groupby(f'split{split_seed}')

    # Save temporary train and test datasets
    for group_name, group_df in grouped:
        filtered = df_all[df_all['CycPeptMPDB_ID'].isin(group_df['CycPeptMPDB_ID'])]
        filtered.to_csv(f'temp_{group_name}.csv', index=False)

    train = pd.read_csv("temp_train.csv", low_memory=False)
    test = pd.read_csv("temp_test.csv", low_memory=False)

    x_train, y_train = preprocess_data(train, drop_cols_range=slice(0, 34))
    x_test, y_test = preprocess_data(test, drop_cols_range=slice(0, 34))

    x_train, x_test = scale_features(x_train, x_test)

    return {'train': [x_train, y_train], 'test': [x_test, y_test], 'test_df': test}


def loader_scaffold_split_scaled(train_list, test_list):
    """
    Loads and processes scaffold split datasets with scaling.

    Parameters:
    - train_list (list of str): List of file paths for training data.
    - test_list (list of str): List of file paths for test data.

    Returns:
    - dict: Processed train and test datasets with DataFrame.
    """
    df_all = pd.read_csv("../CSV/Data/CycPeptMPDB_Peptide_All.csv", low_memory=False)

    # Load and concatenate training datasets
    dfs = [pd.read_csv(file) for file in train_list]
    train_df = pd.concat(dfs, ignore_index=True)

    # Load and concatenate test datasets
    dfs = [pd.read_csv(file) for file in test_list]
    test_df = pd.concat(dfs, ignore_index=True)

    # Filter main dataset based on IDs
    train = df_all[df_all['CycPeptMPDB_ID'].isin(train_df['CycPeptMPDB_ID'])]
    test = df_all[df_all['CycPeptMPDB_ID'].isin(test_df['CycPeptMPDB_ID'])]

    x_train, y_train = preprocess_data(train, drop_cols_range=slice(0, 34))
    x_test, y_test = preprocess_data(test, drop_cols_range=slice(0, 34))

    x_train, x_test = scale_features(x_train, x_test)

    return {'train': [x_train, y_train], 'test': [x_test, y_test], 'test_df': test}
