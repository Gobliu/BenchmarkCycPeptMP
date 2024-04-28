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


def scaffold_split(csv_path, frac_train=0.6, frac_valid=0.2, frac_test=0.2):

    for csv in csv_path:
        df = pd.read_csv(csv)
        df['Scaffold'] = None
        print(len(df.SMILES.tolist()))
        # create dict of the form {scaffold_i: [idx1, idx....]}
        all_scaffolds = {}
        # for i, smiles in enumerate(df.SMILES.tolist()):
        #     scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
        #     # print(i, scaffold)
        #     if scaffold not in all_scaffolds:
        #         all_scaffolds[scaffold] = [i]
        #     else:
        #         all_scaffolds[scaffold].append(i)

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
        # for key, value in all_scaffolds.items():
        #     print(key, value)
        # all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        # print(all_scaffolds)
        all_scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]
        print(all_scaffold_sets)

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

        # split_col = list(range(len(df)))
        # for index in train_idx:
        #     split_col[index] = "train"
        # for index in valid_idx:
        #     split_col[index] = "valid"
        # for index in test_idx:
        #     split_col[index] = "test"

        train_df = df.iloc[train_idx]
        train_df.to_csv(f"{csv[:-4]}_train.csv", index=False)
        valid_df = df.iloc[valid_idx]
        valid_df.to_csv(f"{csv[:-4]}_valid.csv", index=False)
        test_df = df.iloc[test_idx]
        test_df.to_csv(f"{csv[:-4]}_test.csv", index=False)

        # print(set(split_col))
        # print(combined_df.iloc[train_idx].Original_Name_in_Source_Literature.tolist())
        # print(len(set(combined_df.Structurally_Unique_ID.tolist())))
        #
        # # change accordingly
        # for name, idx in zip(["train", "valid", "test"], [train_idx, valid_idx, test_idx]):
        #     traj_combiner_hex_only(ip_dir, combined_df.iloc[idx].Original_Name_in_Source_Literature.tolist(),
        #                            op_dir / f"ss_4in1_{name}")
        #
        # combined_df['Split'] = split_col
        # columns = combined_df.columns.tolist()
        # columns.insert(4, 'Split')
        # df_ = combined_df[columns]
        # df_.to_csv(op_path, index=False)
    #
    # else:
    #     blk_list = [[] for _ in range(blk)]
    #     for i, scaffold_set in enumerate(all_scaffold_sets):
    #         r = i % (2 * blk)           # using the zigzag pattern, hence the pattern length is 2*blk
    #         idx = min(r, 2 * blk - r - 1)
    #         print(i, r, idx, 2 * blk - r - 1)
    #         blk_list[idx].extend(scaffold_set)
    #
    #     split_col = list(range(len(combined_df)))
    #     for i, index_list in enumerate(blk_list):
    #         for index in index_list:
    #             split_col[index] = i
    #
    #     print(split_col)
    #
    #     for i, idx in enumerate(blk_list):
    #         traj_combiner_hex_only(ip_dir, combined_df.iloc[idx].Original_Name_in_Source_Literature.tolist(),
    #                                op_dir / f"scaffold_split_blk{i}")
    #
    #     combined_df['Block'] = split_col
    #     columns = combined_df.columns.tolist()
    #     columns.insert(4, 'Block')
    #     df_ = combined_df[columns]
    #     df_.to_csv(op_path, index=False)


if __name__ == '__main__':
    path = '../CSV/4in1.csv'
    col_list = ['CycPeptMPDB_ID', 'Source', 'Year', 'Original_Name_in_Source_Literature', 'Structurally_Unique_ID',
                'SMILES', 'Molecule_Shape', 'Monomer_Length', 'PAMPA', 'Caco2', 'MDCK', 'RRCK', 'MolLogP']
    clean_df = column_clean(path, col_list)
    mol_length_split(clean_df)
    len_list = [6, 7, 10]
    csv_list = [f"../CSV/mol_length_{i}.csv" for i in len_list]
    scaffold_split(csv_list)
