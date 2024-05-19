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
from MoleculeNet_Main import data_loader, trainer


def split_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=[])
    grouped = df.groupby('Block')
    for group_name, group_df in grouped:
        group_df.to_csv(f'Block_{group_name}.csv', index=False)
    print("Groups saved to CSV files.")


def main(m_name_list, op_dir, seed_list):
    rms = dc.metrics.Metric(dc.metrics.score_function.rms_score)

    blk_size = 10
    for idx in range(blk_size):
        for m_name in m_name_list:
            feat, model = generate_model_feature(m_name, op_dir, batch_size=batch_size)
            tasks, datasets_sol, transformers_sol = dc.molnet.load_delaney(featurizer=feat, splitter='random')
            train_sol, valid_sol, test_sol = datasets_sol

            loader = dc.data.CSVLoader(tasks=['PAMPA'],
                                       feature_field="SMILES",
                                       id_field="Original_Name_in_Source_Literature",
                                       featurizer=feat)
            test_idx = idx
            valid_idx = (idx + 1) % blk_size
            idx_list = [x for x in range(blk_size) if x not in [valid_idx, test_idx]]
            train_list = [f"./CSV/mol_length_6_blk{i}.csv" for i in idx_list]
            valid_list = [f"./CSV/mol_length_6_blk{valid_idx}.csv"]
            test_list = [f"./CSV/mol_length_6_blk{test_idx}.csv"]
            data = data_loader(loader, train_list, valid_list, test_list)
            train_cp, valid_cp, test_cp = data['train'], data['valid'], data['test']
            df = pd.read_csv(f"./CSV/mol_length_6_blk{test_idx}.csv")

            for actual_seed in seed_list:
                manual_seed(actual_seed)
                # ======= pre train
                print(f'==== train solubility data with model {m_name}')

                model = trainer(model, n_epoch=n_epoch_, patience=patience_, train_data=train_sol,
                                valid_data=valid_sol, metrics=[rms], transformers=transformers_sol,
                                text=f'Training solubility with seed {actual_seed}')

                # ======= train
                print(f'==== train cyclic peptide data with model {m_name}')
                transformer_cp = data['transformer']
                model = trainer(model, n_epoch=n_epoch_, patience=patience_, train_data=train_cp,
                                valid_data=valid_cp, metrics=[rms], transformers=[transformer_cp],
                                text=f'Training permeability with seed {actual_seed}')

                # model.reload()
                print('Confirm valid loss', model.evaluate(valid_cp, [rms], transformers=[transformer_cp])['rms_score'])
                print('Test loss', model.evaluate(test_cp, [rms], transformers=[transformer_cp])['rms_score'])
                test_pred = model.predict(test_cp, transformers=[transformer_cp])
                print('RMSE for test data', mean_squared_error(test_pred,
                                                               transformer_cp.untransform(test_cp.y), squared=False))
                print('MAE for test data', mean_absolute_error(test_pred,
                                                               transformer_cp.untransform(test_cp.y)))

                df[f'Pred_{actual_seed}'] = test_pred
                df[f'True_{actual_seed}'] = transformer_cp.untransform(test_cp.y)

            df.to_csv(f'./CSV/{m_name}_block{idx}.csv', index=False)


if __name__ == '__main__':
    n_epoch_ = 20000
    batch_size = 64
    patience_ = 200

    seed_list_ = [123*3**i for i in range(1)]
    model_dir = "../SavedModel/DeepChemPermeability/"
    model_list = ['DMPNN', 'GCN', 'GAT', 'MPNN', 'PAGTN', 'AttentiveFP']
    main(model_list, model_dir, seed_list_)
