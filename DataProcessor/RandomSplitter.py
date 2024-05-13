import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def random_splitter(df, repeat_split=10):
    print('with all length', len(df))
    df = df[df['Monomer_Length'].isin([6, 7, 10])]
    value_counts = {}
    # Iterate through the column
    for value in df['Structurally_Unique_ID']:
        # Increment count for each value
        value_counts[value] = value_counts.get(value, 0) + 1

    # Get values that appeared more than once
    repeated_values = [value for value, count in value_counts.items() if count > 1]

    # Filter the DataFrame based on repeated values
    duplicated_df = df[df['Structurally_Unique_ID'].isin(repeated_values)]
    unique_df = df[~df['Structurally_Unique_ID'].isin(repeated_values)]
    print(len(df), len(duplicated_df), len(unique_df))

    for i in range(repeat_split):
        # Split data into train (80%) and temp (20%)
        train_df, temp_df = train_test_split(unique_df, test_size=0.2, random_state=i)

        # Split temp into validation (50%) and test (50%)
        valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=i)

        # print(len(train_df), len(temp_df), len(valid_df), len(test_df))
        # Print sizes of train, validation, and test sets
        print("Train size:", len(train_df))
        print("Validation size:", len(valid_df))
        print("Test size:", len(test_df))

        # Add dataset type column
        duplicated_df.loc[:, f'split{i}'] = 'train'
        train_df.loc[:, f'split{i}'] = 'train'
        valid_df.loc[:, f'split{i}'] = 'valid'
        test_df.loc[:, f'split{i}'] = 'test'

        # Concatenate DataFrames
        unique_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    # Save combined DataFrame to CSV
    combined_df = pd.concat([duplicated_df, unique_df], ignore_index=True)
    combined_df['Normalized_PAMPA'] = combined_df['PAMPA'].clip(lower=-8)
    combined_df['Normalized_PAMPA'] = (combined_df['Normalized_PAMPA'] + 6) / 2
    combined_df['Binary'] = combined_df['Normalized_PAMPA'].apply(lambda x: 1 if x > 0 else 0)
    combined_df['Soft_Label'] = (combined_df['Normalized_PAMPA'] + 1) / 2
    combined_df['Soft_Label'] = combined_df['Soft_Label'].apply(lambda x: 1 if x > 0.625 else 0 if x < 0.375 else x)
    combined_df.to_csv('../CSV/Data/Random_Split{i}.csv', index=False)

    print(len(unique_df))


if __name__ == '__main__':
    path = '../CSV/Data/CycPeptMPDB_Peptide_All.csv'
    col_list = ['CycPeptMPDB_ID', 'Source', 'Year', 'Original_Name_in_Source_Literature', 'Structurally_Unique_ID',
                'SMILES', 'Molecule_Shape', 'Monomer_Length', 'PAMPA', 'Caco2', 'MDCK', 'RRCK', 'MolLogP']
    clean_df = pd.read_csv(path, usecols=col_list)
    clean_df.dropna(subset=['PAMPA'], inplace=True)
    random_splitter(clean_df)
