import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from deepchem.feat.smiles_tokenizer import SmilesTokenizer


class DataSetSMILES(Dataset):
    """
    A custom PyTorch Dataset for handling SMILES data.

    This dataset is designed to read multiple CSV files containing SMILES data, tokenize the SMILES strings,
    and provide a mechanism to assign class weights based on the distribution of binary labels.

    Attributes:
        df (pd.DataFrame): A DataFrame containing the dataset loaded from CSV files.
        tokenizer (SmilesTokenizer): The tokenizer used for encoding SMILES strings.
        x_column (str): The name of the column containing input features (SMILES strings).
        y_column (str): The name of the column containing target labels.
        id_column (str): The name of the column containing unique identifiers.
    """

    def __init__(self, csv_path_list, dictionary_path, x_column, y_columns, id_column='CycPeptMPDB_ID'):
        """
        Initializes the dataset by loading data from the given list of CSV file paths and setting up the tokenizer.

        Args:
            csv_path_list (list of str): List of paths to CSV files.
            dictionary_path (str): Path to the dictionary used for SMILES tokenization.
            x_column (str): Column name for input features (SMILES strings).
            y_column (str): Column name for target labels.
            id_column (str): Column name for unique identifiers. Defaults to 'CycPeptMPDB_ID'.

        Raises:
            ValueError: If any of the required columns are missing from the DataFrame.
        """
        df_list = []
        for f in csv_path_list:
            df_list.append(pd.read_csv(f))
        self.df = pd.concat(df_list, ignore_index=True)
        self.tokenizer = SmilesTokenizer(dictionary_path)
        # self.vocab = self.load_vocabulary(dictionary_path)
        assert isinstance(x_column, str), 'make sure x_column is a string, not a list'
        self.x_column = x_column
        self.y_columns = y_columns
        self.id_column = id_column
        self.df['weight'] = np.ones(len(self.df))

        # Verify that required columns exist in the DataFrame
        required_columns = [x_column, id_column, 'Binary'] + self.y_columns
        print(required_columns)
        # print(self.df.columns)
        for col in required_columns:
            print(col, self.df.columns)
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        print(f'Sample size: {self.__len__()}')
        print(self.df.SMILES)

    def get_weight(self):
        """
        Computes class weights for positive and negative samples based on the 'Binary' column.

        The weights are calculated as the inverse of the class frequency, normalized by the total number of samples.

        Returns:
            tuple: A tuple containing positive weight (pw) and negative weight (nw).

        Raises:
            ValueError: If the 'Binary' column is missing, or if there are no positive or negative samples.
        """
        if 'Binary' not in self.df.columns:
            raise ValueError("The DataFrame must contain a 'Binary' column for weight calculation.")

        total = len(self.df)
        positive = sum(self.df.Binary)
        negative = total - positive

        if positive == 0 or negative == 0:
            raise ValueError(f"Cannot calculate weights: {positive} positive samples, {negative} negative samples.")

        print(f'In total {total} samples in train data, {positive} positive, {negative} negative')
        pw = total / (positive * 2)
        nw = total / (negative * 2)
        print(f'Positive weight {pw}, Negative weight {nw}')
        return pw, nw

    def set_weight(self, pos_weight, neg_weight):
        """
        Sets the weights for each sample in the DataFrame based on the 'Binary' column.

        Args:
            pos_weight (float): The weight for positive samples (Binary >= 0.5).
            neg_weight (float): The weight for negative samples (Binary < 0.5).

        Raises:
            ValueError: If the 'Binary' column is missing from the DataFrame.
        """
        if 'Binary' not in self.df.columns:
            raise ValueError("The DataFrame must contain a 'Binary' column to set weights.")

        self.df.loc[self.df.Binary >= 0.5, 'weight'] = pos_weight
        self.df.loc[self.df.Binary < 0.5, 'weight'] = neg_weight

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the following:
                - id_ (str): The unique identifier of the sample.
                - x (torch.Tensor): The tokenized SMILES string, padded to length 64.
                - y (torch.Tensor): The target label.
                - w (float): The weight of the sample.
                - sample_dict (dict): A dictionary representation of the sample's DataFrame row.

        Raises:
            ValueError: If the index is out of bounds, or if token length exceeds 64.
        """
        if idx >= self.__len__():
            raise ValueError(f'Index {idx} exceeds the length of the dataset: {self.__len__()}')

        try:
            x = str(self.df.loc[idx, self.x_column])
            # print(len(x), type(x), x)
            # print(self.smi_tokenizer(x))
            # print(self.convert_tokens_to_ids(self.smi_tokenizer(x)))
            x = self.tokenizer.encode(x)
            # print(len(x), x)
        except Exception as e:
            raise ValueError(f"Tokenization error at index {idx}: {str(e)}")

        y = torch.tensor(self.df.loc[idx, self.y_columns])
        w = self.df.iloc[idx]['weight']  # Using .iloc to avoid SettingWithCopyWarning
        id_ = self.df.loc[idx, self.id_column]

        if len(x) <= 180:
            x += [0] * (180 - len(x))
        else:
            # print(len(x), x)
            raise ValueError('Token length is larger than 180.')

        x = torch.tensor(x, dtype=torch.int)
        return id_, x, y, w, self.df.loc[idx].to_dict()  # Include weight in the return value

    # @staticmethod
    # def smi_tokenizer(smi):
    #     """
    #     Tokenize a SMILES molecule or reaction
    #     """
    #     import re
    #     pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    #     regex = re.compile(pattern)
    #     # print(type(smi), smi)
    #     tokens = [token for token in regex.findall(smi)]
    #     assert smi == ''.join(tokens)
    #     return ' '.join(tokens)
    #
    # @staticmethod
    # def load_vocabulary(vocab_file):
    #     with open(vocab_file, 'r') as f:
    #         vocab = [line.strip() for line in f if line.strip()]
    #     token_to_id = {token: idx for idx, token in enumerate(vocab)}
    #     return token_to_id
    #
    # # Generated by ChatGPT, has bug
    # # Convert tokenized SMILES to token IDs
    # def convert_tokens_to_ids(self, tokens):
    #     token_ids = [self.vocab[token] for token in tokens if token in self.vocab]
    #     return token_ids
