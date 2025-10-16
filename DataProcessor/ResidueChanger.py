import pandas as pd


def mutator(df, src_str, tgt_str, output_path):
    # Add a new column for mutated SMILES
    df['mutant'] = ''

    # Loop through each row
    for i, row in df.iterrows():
        smiles = row['SMILES']
        if src_str in smiles:
            # Replace only the first match of src_str with tgt_str
            new_smiles = smiles.replace(src_str, tgt_str)
            df.at[i, 'mutant'] = new_smiles  # store result in new column

    # Remove rows without mutation (empty mutant field)
    df = df[df['mutant'] != '']

    # Save mutated dataset
    df.to_csv(output_path, index=False)
    print(f"✅ Saved mutated dataset with {len(df)} entries to {output_path}")


if __name__ == '__main__':
    path = '../CSV/Data/Random_Split.csv'
    col_list = ['CycPeptMPDB_ID', 'Source', 'Year', 'Original_Name_in_Source_Literature',
                'Structurally_Unique_ID', 'SMILES', 'Molecule_Shape', 'Monomer_Length', 'PAMPA', 'Normalized_PAMPA']

    clean_df = pd.read_csv(path, usecols=col_list)
    clean_df.dropna(subset=['PAMPA'], inplace=True)
    clean_df = clean_df[clean_df['Monomer_Length'] != 10]  # fixed boolean indexing

    mutator(clean_df, src_str='CC(C)C', tgt_str='CC(C)=CCC', output_path='../CSV/Data/LEU2PBL.csv')
