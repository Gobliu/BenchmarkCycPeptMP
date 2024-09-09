import pandas as pd
from sklearn.preprocessing import StandardScaler


def loader_random_split_scaled(split_seed):
    df_all = pd.read_csv("../CSV/Data/CycPeptMPDB_Peptide_All.csv", low_memory=False)
    df_random_split = pd.read_csv("../CSV/Data/Random_Split.csv")
    grouped = df_random_split.groupby(f'split{split_seed}')
    for group_name, group_df in grouped:
        filtered = df_all[df_all['CycPeptMPDB_ID'].isin(group_df['CycPeptMPDB_ID'])]
        filtered.to_csv(f'temp_{group_name}.csv', index=False)

    train = pd.read_csv("temp_train.csv", low_memory=False)
    x_train = train.drop(train.columns[0:34], axis=1)
    y_train = train['Permeability']

    test = pd.read_csv("temp_test.csv", low_memory=False)
    x_test = test.drop(train.columns[0:34], axis=1)
    y_test = test['Permeability']

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return {'train': [x_train, y_train], 'test': [x_test, y_test], 'test_df': test}


def loader_scaffold_split_scaled(train_list, test_list):
    df_all = pd.read_csv("../CSV/Data/CycPeptMPDB_Peptide_All.csv", low_memory=False)

    dfs = [pd.read_csv(file) for file in train_list]
    dfs = pd.concat(dfs, ignore_index=True)
    train = df_all[df_all['CycPeptMPDB_ID'].isin(dfs['CycPeptMPDB_ID'])]
    x_train = train.drop(train.columns[0:34], axis=1)
    y_train = train['Permeability']

    dfs = [pd.read_csv(file) for file in test_list]
    dfs = pd.concat(dfs, ignore_index=True)
    test = df_all[df_all['CycPeptMPDB_ID'].isin(dfs['CycPeptMPDB_ID'])]
    x_test = test.drop(train.columns[0:34], axis=1)
    y_test = test['Permeability']

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # print(train)
    # print(test)
    return {'train': [x_train, y_train], 'test': [x_test, y_test], 'test_df': test}
