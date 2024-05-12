import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# from ScaffoldSpliter import column_clean


def duplicate_checker(df):
    value_counts = {}
    # Iterate through the column
    for value in df['Structurally_Unique_ID']:
        # Increment count for each value
        value_counts[value] = value_counts.get(value, 0) + 1

    # Get values that appeared more than once
    repeated_values = [value for value, count in value_counts.items() if count > 1]
    print(len(value_counts), len(repeated_values))

    # Filter the DataFrame based on repeated values
    filtered_df = df[df['Structurally_Unique_ID'].isin(repeated_values)]
    print(filtered_df)
    filtered_df.to_csv('Duplicated_PAMPA', index=False)


def duplicate_error_calculator(csv_path):
    df = pd.read_csv(csv_path)
    uid_list = set(df.Structurally_Unique_ID)
    print('uid_list', uid_list, len(uid_list), len(df))
    ae_list = []
    for uid in uid_list:
        filtered_df = df[df['Structurally_Unique_ID'] == uid]
        pampa_list = filtered_df.PAMPA.tolist()
        print('uid', uid)
        print(filtered_df)
        print(pampa_list)
        for i in range(len(pampa_list) - 1):
            for j in range(i + 1, len(pampa_list)):  # Exclude self-differences by starting from i+1
                ae_list.append(abs(pampa_list[i] - pampa_list[j]))

    mae = np.mean(ae_list)
    std = np.std(ae_list)
    print(mae, std)


def compare_diff_experiments(csv_path, exp_list):
    df = pd.read_csv(csv_path)
    for i, exp1 in enumerate(exp_list):
        # print(i, exp1)
        for exp2 in exp_list[i+1:]:
            clone_df = df.copy()
            clone_df.dropna(subset=[exp1, exp2], inplace=True)
            clone_df.to_csv(f'{exp1}_{exp2}.csv', index=False)
            error = clone_df[exp1] - clone_df[exp2]
            mean = np.mean(np.abs(error))
            std = np.std(np.abs(error))
            print(exp1, exp2, mean, std, len(clone_df))


def three_repeats(csv_path):
    df = pd.read_csv(csv_path)
    print(df)
    # df = df[~(df == 0).any(axis=1)]
    # print(df)
    print(df.columns)
    print(np.abs(np.log10(df.Papp1)))


if __name__ == '__main__':
    path = '../CSV/CycPeptMPDB_Peptide_All.csv'
    col_list = ['CycPeptMPDB_ID', 'Source', 'Year', 'Original_Name_in_Source_Literature', 'Structurally_Unique_ID',
                'SMILES', 'Molecule_Shape', 'Monomer_Length', 'PAMPA', 'Caco2', 'MDCK', 'RRCK', 'MolLogP']
    clean_df = pd.read_csv(path, usecols=col_list)
    # clean_df.dropna(subset=['PAMPA'], inplace=True)
    # duplicate_checker(clean_df)
    # duplicate_error_calculator('Duplicated_PAMPA_inter.csv')
    # compare_diff_experiments(path, exp_list=['PAMPA', 'Caco2', 'MDCK', 'RRCK'])
    three_repeats('./Furukawa_3repeats.csv')
