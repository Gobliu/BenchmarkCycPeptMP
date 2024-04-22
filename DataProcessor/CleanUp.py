import pandas as pd


def column_cleaner(csv_path, column_list):
    df = pd.read_csv(csv_path, usecols=column_list)
    return df


def mol_length_spliter(df):
    grouped = df.groupby('Monomer_Length')
    # Iterate over the groups and save each one to a separate CSV file
    for name, group in grouped:
        filename = f"../CSV/mol_length_{name}.csv"  # Generate a filename based on the group name
        group.to_csv(filename, index=False)  # Save the group to a CSV file without including the index
        print(f"Group {name} has {len(group)} samples, saved to '{filename}'")


if __name__ == '__main__':
    path = '../CSV/CycPeptMPDB_Peptide_All.csv'
    col_list = ['CycPeptMPDB_ID', 'Source', 'Year', 'Original_Name_in_Source_Literature', 'Structurally_Unique_ID',
                'SMILES', 'Molecule_Shape', 'Monomer_Length', 'PAMPA', 'Caco2', 'MDCK', 'RRCK', 'MolLogP']
    clean_df = column_cleaner(path, col_list)
    mol_length_spliter(clean_df)