import os
import sys
import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_squared_error, mean_absolute_error

import deepchem as dc
import deepchem.models.losses as losses
import pandas as pd
import torch.nn as nn

from Utils import manual_seed
from ModelFeatureGenerator import generate_model_feature
from Trainer import trainer


def data_loader(ip_df, split_seed, loader):
    grouped = ip_df.groupby(f'split{split_seed}')
    for group_name, group_df in grouped:
        group_df.to_csv(f'temp_{group_name}.csv', index=False)
    train = loader.create_dataset('temp_train.csv')
    valid = loader.create_dataset('temp_valid.csv')
    test = loader.create_dataset('temp_test.csv')
    print(f'Got {len(train.y)} samples for train, {len(valid.y)} for valid, and {len(test.y)} for test')
    return {'train': train, 'valid': valid, 'test': test}


def main(ip_path, op_dir, m_name):
    rms = dc.metrics.Metric(dc.metrics.score_function.rms_score)
    ip_df = pd.read_csv(ip_path)

    for split_seed in range(10):
        manual_seed(123 * split_seed ** 2)
        feat, model = generate_model_feature(m_name, op_dir, batch_size=batch_size, mode=mode)
        tasks, datasets_sol, transformers_sol = dc.molnet.load_delaney(featurizer=feat, splitter='random')
        train_sol, valid_sol, test_sol = datasets_sol
        loader = dc.data.CSVLoader(tasks=[task],
                                   feature_field="SMILES",
                                   id_field="Original_Name_in_Source_Literature",
                                   featurizer=feat)

        data = data_loader(ip_df, split_seed, loader)
        train_cp, valid_cp, test_cp = data['train'], data['valid'], data['test']

        # ======= pre train
        print('==== train solubility data')

        model = trainer(model, n_epoch=n_epoch // 5, patience=patience, train_data=train_sol,
                        valid_data=valid_sol, metrics=[rms], transformers=transformers_sol,
                        text=f'Training solubility with split seed {split_seed}')

        df = pd.read_csv('temp_test.csv')
        # ======= train
        print('==== train cyclic peptide data')
        model = trainer(model, n_epoch=n_epoch, patience=patience, train_data=train_cp,
                        valid_data=valid_cp, metrics=[rms], transformers=[],
                        text=f'Training permeability with split seed {split_seed}')

        print('Confirm valid loss', model.evaluate(valid_cp, [rms])['rms_score'])
        test_pred = model.predict(test_cp)
        print('RMSE for test data', mean_squared_error(model.predict(test_cp), test_cp.y, squared=False))
        print('MAE for test data', mean_absolute_error(test_pred, test_cp.y))
        print('test pred rmse:', np.mean((model.predict(test_cp) - test_cp.y)**2)**0.5)
        df[f'Pred_{split_seed}'] = test_pred
        df[f'True_{split_seed}'] = test_cp.y
        model.save_checkpoint()

        df.to_csv(f'./CSV/Predictions/TVT_Random_Split/{m_name}_SplitSeed{split_seed}.csv', index=False)


if __name__ == '__main__':
    n_epoch = 20000
    batch_size = 64
    patience = 200
    task = "Normalized_PAMPA"
    mode = "regression"
    model_dir = "../SavedModel/DeepChemPermeability/TVT_split/"

    # model_list = ['DMPNN', 'GCN', 'GAT', 'MPNN', 'PAGTN', 'AttentiveFP']
    csv_path = "./CSV/Data/Random_Split.csv"
    main(csv_path, model_dir, m_name="MPNN")

