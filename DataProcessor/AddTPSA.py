import pandas as pd

mol_length_list = [6, 7, 10]
csv_list = (
    [f'mol_length_{i}_train.csv' for i in mol_length_list] +
    [f'mol_length_{i}_valid.csv' for i in mol_length_list] +
    [f'mol_length_{i}_test.csv' for i in mol_length_list] +
    ["Random_Split.csv"]
)

for csv_name in csv_list:

    # Load the CSV files
    random_split_df = pd.read_csv(f'../CSV/Data/{csv_name}')
    cycpeptmpdb_df = pd.read_csv('../CSV/Data/CycPeptMPDB_Peptide_All.csv', low_memory=False)

    # Merge the DataFrames on 'CycPeptMPDB_ID' and select relevant columns
    merged_df = pd.merge(random_split_df, cycpeptmpdb_df[['CycPeptMPDB_ID', 'TPSA', 'Sequence_TPSA']],
                         on='CycPeptMPDB_ID', how='left')

    merged_df['Sequence_TPSA'] = merged_df['Sequence_TPSA'].apply(eval)
    Sum_Seq_PSA = []
    TPSA_consistence = []
    for row in merged_df.itertuples(index=False):
        Sum_PSA = sum(row.Sequence_TPSA)
        Sum_Seq_PSA.append(Sum_PSA)
        if abs(Sum_PSA - row.TPSA) < 0.01:
            TPSA_consistence.append('TRUE')
        else:
            TPSA_consistence.append('TRUE')
    merged_df['Sum_Seq_PSA'] = Sum_Seq_PSA
    merged_df['TPSA_consistence'] = TPSA_consistence

    random_split_df.to_csv(f'../CSV/Data/{csv_name}', index=False)

