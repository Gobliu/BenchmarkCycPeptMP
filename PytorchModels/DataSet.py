import torch
from torch.utils.data import Dataset
import pandas as pd
from deepchem.feat.smiles_tokenizer import SmilesTokenizer


class DataSetSMILES(Dataset):
    def __init__(self, csv_path_list, dictionary_path, x_column, y_column, id_column='CycPeptMPDB_ID'):
        df_list = []
        for f in csv_path_list:
            df_list.append(pd.read_csv(f))
        self.df = pd.concat(df_list, ignore_index=True)
        self.tokenizer = SmilesTokenizer(dictionary_path)
        self.x_column = x_column
        self.y_column = y_column
        self.id_column = id_column

        print(f'Sample size', self.__len__())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise ValueError(f'Index {idx} exceeds the length of the dataset: {self.__len__()}')

        x = self.df.loc[idx, self.x_column].to_string(index=False, header=False)
        x = self.tokenizer.encode(x)
        y = torch.tensor(self.df.loc[idx, self.y_column])
        id_ = self.df.loc[idx, self.id_column]
        if len(x) <= 64:
            x += [0] * (64-len(x))
        else:
            quit('Token length larger than 64')
        # print('id', id_, 'x', x, len(x), 'y', y)
        x = torch.tensor(x, dtype=torch.int)
        return id_, x, y, self.df.loc[idx].to_dict()
