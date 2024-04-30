def scaffold_split(csv_list, op_path, blk, frac_train=0.8, frac_valid=0.1, frac_test=0.1):

    # Initialize an empty list to store DataFrames
    dfs = []

    # Read each CSV file into a DataFrame and append it to the list
    for file in csv_list:
        df = pd.read_csv(file)
        dfs.append(df)

    # Concatenate the list of DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    print(len(combined_df.SMILES.tolist()))

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in enumerate(combined_df.SMILES.tolist()):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
        # print(i, scaffold)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    print(len(all_scaffolds))
    print(all_scaffolds)
    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    print(all_scaffolds)
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    print(all_scaffold_sets)

    ip_dir = Path("../../Data/DrugPermeability/traj_npz/")
    op_dir = Path("../../Data/DrugPermeability/traj_npz_blk_hex_only/")

    # get train, valid test indices
    if blk is None:
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        train_cutoff = frac_train * len(combined_df)
        valid_cutoff = (frac_train + frac_valid) * len(combined_df)
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(train_idx))) == 0
        assert len(combined_df) == len(set(train_idx)) + len(set(valid_idx)) + len(set(test_idx))
        print(sorted(test_idx))

        split_col = list(range(len(combined_df)))
        for index in train_idx:
            split_col[index] = "train"
        for index in valid_idx:
            split_col[index] = "valid"
        for index in test_idx:
            split_col[index] = "test"

        print(set(split_col))
        print(combined_df.iloc[train_idx].Original_Name_in_Source_Literature.tolist())
        print(len(set(combined_df.Structurally_Unique_ID.tolist())))

        # change accordingly
        for name, idx in zip(["train", "valid", "test"], [train_idx, valid_idx, test_idx]):
            traj_combiner_hex_only(ip_dir, combined_df.iloc[idx].Original_Name_in_Source_Literature.tolist(),
                                   op_dir / f"ss_4in1_{name}")

        combined_df['Split'] = split_col
        columns = combined_df.columns.tolist()
        columns.insert(4, 'Split')
        df_ = combined_df[columns]
        df_.to_csv(op_path, index=False)

    else:
        blk_list = [[] for _ in range(blk)]
        for i, scaffold_set in enumerate(all_scaffold_sets):
            r = i % (2 * blk)           # using the zigzag pattern, hence the pattern length is 2*blk
            idx = min(r, 2 * blk - r - 1)
            print(i, r, idx, 2 * blk - r - 1)
            blk_list[idx].extend(scaffold_set)

        split_col = list(range(len(combined_df)))
        for i, index_list in enumerate(blk_list):
            for index in index_list:
                split_col[index] = i

        print(split_col)

        for i, idx in enumerate(blk_list):
            traj_combiner_hex_only(ip_dir, combined_df.iloc[idx].Original_Name_in_Source_Literature.tolist(),
                                   op_dir / f"scaffold_split_blk{i}")

        combined_df['Block'] = split_col
        columns = combined_df.columns.tolist()
        columns.insert(4, 'Block')
        df_ = combined_df[columns]
        df_.to_csv(op_path, index=False)