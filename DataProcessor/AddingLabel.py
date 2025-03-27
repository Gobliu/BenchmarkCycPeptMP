import pandas as pd


def adding_label(df):
    """
    Adds normalized and binary labels based on the 'PAMPA' column.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with a 'PAMPA' column.

    Returns:
    - pd.DataFrame: DataFrame with added label columns.
    """
    if 'PAMPA' not in df:
        raise ValueError("Error: The input DataFrame must contain a 'PAMPA' column.")

    df['Normalized_PAMPA'] = df['PAMPA'].clip(lower=-8)
    df['Normalized_PAMPA'] = (df['Normalized_PAMPA'] + 6) / 2

    # The line below behaves the same as: df['Binary'] = df['Normalized_PAMPA'].apply(lambda x: 1 if x > 0 else 0)
    # It creates a boolean mask where values > 0 are True (1), and others are False (0), then converts to integers.
    df['Binary'] = (df['Normalized_PAMPA'] > 0).astype(int)

    df['Soft_Label'] = (df['Normalized_PAMPA'] + 1) / 2
    df['Soft_Label'] = df['Soft_Label'].apply(lambda x: 1 if x > 0.625 else 0 if x < 0.375 else x)

    return df


def load_and_filter_data(path, monomer_length):
    """
    Loads a CSV file, filters by monomer length, and drops NaN values in 'PAMPA'.

    Parameters:
    - path (str): File path to the CSV.
    - monomer_length (int): The monomer length to filter by.

    Returns:
    - pd.DataFrame: Filtered DataFrame.

    Raises:
    - FileNotFoundError: If the CSV file is missing.
    - ValueError: If there's an issue reading the CSV file.
    """
    col_list = [
        'CycPeptMPDB_ID', 'Source', 'Year', 'Original_Name_in_Source_Literature',
        'Structurally_Unique_ID', 'SMILES', 'Molecule_Shape', 'Monomer_Length',
        'PAMPA', 'Caco2', 'MDCK', 'RRCK', 'MolLogP'
    ]

    try:
        df = pd.read_csv(path, usecols=col_list)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found - {path}")
    except Exception as e:
        raise ValueError(f"Error reading file {path}: {e}")

    # Ensure 'Monomer_Length' column exists
    if 'Monomer_Length' not in df:
        raise ValueError(f"Error: The 'Monomer_Length' column is missing in {path}.")

    # Filter by monomer length
    df = df[df['Monomer_Length'] == monomer_length]

    # Ensure 'PAMPA' column exists
    if 'PAMPA' not in df:
        raise ValueError(f"Error: The 'PAMPA' column is missing in {path}.")

    df.dropna(subset=['PAMPA'], inplace=True)

    if df.empty:
        raise ValueError(f"Error: No data left after filtering by Monomer Length {monomer_length} in {path}.")

    print(f"Filtered data for Monomer Length {monomer_length}: {len(df)} records found.")
    return df


if __name__ == '__main__':
    path = '../CSV/Data/CycPeptMPDB_Peptide_All.csv'

    for monomer_length in [2, 3, 4, 5, 8, 9, 11, 12, 13, 14, 15]:
        df = load_and_filter_data(path, monomer_length)

        df = adding_label(df)

        output_path = f'../CSV/Data/mol_length_{monomer_length}.csv'
        df.to_csv(output_path, index=False)
        print(f"Saved filtered dataset to: {output_path}")
