import pandas as pd

mol_length_list = [6, 7, 10]
# csv_list = (
#     [f'mol_length_{i}_train.csv' for i in mol_length_list] +
#     [f'mol_length_{i}_valid.csv' for i in mol_length_list] +
#     [f'mol_length_{i}_test.csv' for i in mol_length_list] +
#     ["Random_Split.csv"]
# )
csv_list = ["Random_Split.csv"]

for csv_name in csv_list:

    df = pd.read_csv(f'../CSV/Data/{csv_name}')

    column_list = ['MolLogP', 'TPSA']

    for col in column_list:

        stats = df[col].agg(['mean', 'std', 'min', 'max'])
        print(f'{col}', stats)
        df[f'Normalized_{col}'] = 2 * (df[col] - df[col].min()) / (df[col].max() - df[col].min()) - 1

    df.to_csv(f'../CSV/Data/{csv_name}', index=False)
