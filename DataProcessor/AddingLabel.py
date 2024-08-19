import pandas as pd


def adding_label(df):
    df['Normalized_PAMPA'] = df['PAMPA'].clip(lower=-8)
    df['Normalized_PAMPA'] = (df['Normalized_PAMPA'] + 6) / 2
    df['Binary'] = df['Normalized_PAMPA'].apply(lambda x: 1 if x > 0 else 0)
    df['Soft_Label'] = (df['Normalized_PAMPA'] + 1) / 2
    df['Soft_Label'] = df['Soft_Label'].apply(lambda x: 1 if x > 0.625 else 0 if x < 0.375 else x)
    return df


if __name__ == '__main__':
    path = '../CSV/Data/CycPeptMPDB_Peptide_All.csv'
    col_list = ['CycPeptMPDB_ID', 'Source', 'Year', 'Original_Name_in_Source_Literature', 'Structurally_Unique_ID',
                'SMILES', 'Molecule_Shape', 'Monomer_Length', 'PAMPA', 'Caco2', 'MDCK', 'RRCK', 'MolLogP']
    for i in [2, 3, 4, 5, 8, 9, 11, 12, 13, 14, 15]:
        df_ = pd.read_csv(path, usecols=col_list)
        print(df_)
        df_ = df_[df_['Monomer_Length'].isin([i])]
        print(f'length {i} only', len(df_))
        df_.dropna(subset=['PAMPA'], inplace=True)
        df_ = adding_label(df_)
        df_.to_csv(f'../CSV/Data/mol_length_{i}.csv', index=False)
