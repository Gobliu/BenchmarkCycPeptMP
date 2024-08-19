import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def clean_assay_type(path, assay_type):
    df = pd.read_csv(path, usecols=col_list)
    df.dropna(subset=assay_type, inplace=True)
    return df


def length_splitter(df):
    groups = df.groupby('Monomer_Length')
    # Iterate through the groups and save each to a separate CSV file
    for group_name, group_df in groups:
        filename = f'{op_dir}mol_length_{group_name}.csv'
        group_df.to_csv(filename, index=False)
        print(f"Group {group_name} has {len(group_df)} samples, saved to '{filename}'")


if __name__ == '__main__':
    csv_path = '../CSV/Data/CycPeptMPDB_Peptide_All.csv'
    op_dir = '../CSV/Data/'
    col_list = ['CycPeptMPDB_ID', 'Source', 'Year', 'Original_Name_in_Source_Literature', 'Structurally_Unique_ID',
                'SMILES', 'Molecule_Shape', 'Monomer_Length', 'PAMPA', 'Caco2', 'MDCK', 'RRCK', 'MolLogP']
    clean_df = clean_assay_type(csv_path, ['PAMPA'])
    length_splitter(clean_df)
