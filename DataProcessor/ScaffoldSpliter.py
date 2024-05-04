import numpy as np
import pandas as pd
from pathlib import Path
from rdkit.Chem.Scaffolds import MurckoScaffold


def column_clean(csv_path, column_list):
    df = pd.read_csv(csv_path, usecols=column_list)
    df.dropna(subset=['PAMPA'], inplace=True)
    return df


def mol_length_split(df):
    grouped = df.groupby('Monomer_Length')
    # Iterate over the groups and save each one to a separate CSV file
    for name, group in grouped:
        filename = f"../CSV/mol_length_{name}.csv"  # Generate a filename based on the group name
        group.to_csv(filename, index=False)  # Save the group to a CSV file without including the index
        print(f"Group {name} has {len(group)} samples, saved to '{filename}'")


def scaffold_cluster(csv_path):
    df = pd.read_csv(csv_path)
    df['Scaffold'] = None
    print(len(df.SMILES.tolist()))
    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}

    for i, row in df.iterrows():
        smiles = row['SMILES']
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
        df.loc[i, 'Scaffold'] = scaffold
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    print(len(all_scaffolds))
    print(all_scaffolds)
    # sort from largest to smallest sets
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    print(all_scaffold_sets)
    return df, all_scaffold_sets


def block_spliter(csv_list, n_blk):
    for csv in csv_list:
        df, all_scaffold_sets = scaffold_cluster(csv)
        blk_list = [[] for _ in range(n_blk)]
        for i, scaffold_set in enumerate(all_scaffold_sets):
            r = i % (2 * n_blk)           # using the zigzag pattern, hence the pattern length is 2*blk
            idx = min(r, 2 * n_blk - r - 1)
            print(i, r, idx, 2 * n_blk - r - 1)
            blk_list[idx].extend(scaffold_set)

        split_col = list(range(len(df)))
        for i, index_list in enumerate(blk_list):
            for index in index_list:
                split_col[index] = i

        for i, blk_indices in enumerate(blk_list):
            blk_df = df.iloc[blk_indices]
            print(len(blk_df))
            blk_df.to_csv(f"../CSV/{csv[:-4]}_blk{i}.csv", index=False)


def train_valid_test_spliter(csv_list, frac_train=0.6, frac_valid=0.2, frac_test=0.2):
    for csv in csv_list:
        df, all_scaffold_sets = scaffold_cluster(csv)
        # get train, valid test indices
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        train_cutoff = frac_train * len(df)
        valid_cutoff = (frac_train + frac_valid) * len(df)
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) < train_cutoff:
                train_idx.extend(scaffold_set)
                print('go to train', len(scaffold_set))
            elif len(train_idx) + len(valid_idx) < valid_cutoff:
                valid_idx.extend(scaffold_set)
                print('go to valid', len(scaffold_set))
            else:
                test_idx.extend(scaffold_set)
                print('go to test', len(scaffold_set))

        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(train_idx))) == 0
        assert len(df) == len(set(train_idx)) + len(set(valid_idx)) + len(set(test_idx))
        # print(sorted(test_idx))

        train_df = df.iloc[train_idx]
        train_df.to_csv(f"../CSV/{csv[:-4]}_train.csv", index=False)
        valid_df = df.iloc[valid_idx]
        valid_df.to_csv(f"../CSV/{csv[:-4]}_valid.csv", index=False)
        test_df = df.iloc[test_idx]
        test_df.to_csv(f"../CSV/{csv[:-4]}_test.csv", index=False)


if __name__ == '__main__':
    path = '../CSV/4in1.csv'
    col_list = ['CycPeptMPDB_ID', 'Source', 'Year', 'Original_Name_in_Source_Literature', 'Structurally_Unique_ID',
                'SMILES', 'Molecule_Shape', 'Monomer_Length', 'PAMPA', 'Caco2', 'MDCK', 'RRCK', 'MolLogP']
    clean_df = column_clean(path, col_list)
    mol_length_split(clean_df)
    len_list = [6, 7, 10]
    csv_list_ = [f"../CSV/mol_length_{i}.csv" for i in len_list]
    # block_spliter(csv_list_, n_blk=10)
    train_valid_test_spliter(csv_list_, frac_train=0.8, frac_valid=0.1, frac_test=0.1)