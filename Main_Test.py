import os
import sys
import numpy as np
# import random
from copy import deepcopy

import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

import deepchem as dc
import deepchem.models.losses as losses
import pandas as pd
import torch.nn as nn

from Utils import set_seed
from ModelFeatureGenerator import generate_model_feature
from ModelTrainer import trainer_regression


def data_loader(train_list, valid_list, test_list, loader):
    train = loader.create_dataset(train_list)
    valid = loader.create_dataset(valid_list)
    test = loader.create_dataset(test_list)
    print(f'Got {len(train.y)} samples for train, {len(valid.y)} for valid, and {len(test.y)} for test')
    return {'train': train, 'valid': valid, 'test': test}


def main(m_name_list, op_dir, seed_list):
    for m_name in m_name_list:
        print(f'==== training {m_name} model')
        df_list = []
        for f in test_list:
            df_list.append(pd.read_csv(f))
        df = pd.concat(df_list, ignore_index=True)
        # print(df.keys())
        rms = dc.metrics.Metric(dc.metrics.score_function.rms_score)
        loss = dc.models.losses.L1Loss
        for actual_seed in seed_list:
            set_seed(actual_seed)
            feat, model = generate_model_feature(m_name, op_dir, batch_size=batch_size, mode=mode)
            print(model.loss)
            tasks_sol, datasets_sol, transformers_sol = dc.molnet.load_delaney(featurizer=feat, splitter='random')
            print(type(datasets_sol))
            # for i in datasets_sol:
            #     print(i)
            print(tasks_sol)
            train_sol, valid_sol, test_sol = datasets_sol
            # print(type(train_sol), train_sol.__dict__)
            # print(train_sol.y.shape, train_sol.w.shape, type(test_sol.y), type(test_sol.w))
            for d in [train_sol, valid_sol, test_sol]:
                print(d.X.shape, d.y.shape, d.w.shape)
                # print(d.ids)

            train_sol = dc.data.NumpyDataset(X=train_sol.X, y=np.repeat(train_sol.y, 2, axis=1),
                                             w=np.repeat(train_sol.w, 2, axis=1),
                                             ids=train_sol.ids)
            valid_sol = dc.data.NumpyDataset(X=valid_sol.X, y=np.repeat(valid_sol.y, 2, axis=1),
                                             w=np.repeat(valid_sol.w, 2, axis=1))
            test_sol = dc.data.NumpyDataset(X=test_sol.X, y=np.repeat(test_sol.y, 2, axis=1),
                                            w=np.repeat(test_sol.w, 2, axis=1))
            # print(model.model)
            # model.loss = loss
            # print(model.loss)
            # print(transformers_sol)
            # for name, param in model.model.named_parameters():
            #     print(name, param.shape, param.numel())
                # para_size += param.numel()
            # model.model.encoder.requires_grad_(False)
            # model.model.ffn = nn.Linear(24, 2)
            # for name, param in model.model.named_parameters():
            #     print(name, param.shape, param.requires_grad)

            # train_sol.y = np.repeat(train_sol.y, 2, axis=1)
            # train_sol.w = np.repeat(train_sol.w, 2, axis=1)
            print(train_sol.y.shape, train_sol.w.shape, type(test_sol.y), type(test_sol.w))
            print('==== train solubility data')

            model = trainer_regression(model, n_epoch=n_epoch // 5, patience=patience, train_data=train_sol,
                            valid_data=valid_sol, metrics=[rms], transformers=transformers_sol,
                            text=f'Training solubility with seed {actual_seed}')

            continue
            loader = dc.data.CSVLoader(
                tasks=[task],
                feature_field="SMILES",
                id_field="Original_Name_in_Source_Literature",
                featurizer=feat)

            data = data_loader(train_list, valid_list, test_list, loader)
            train_cp, valid_cp, test_cp = data['train'], data['valid'], data['test']
            # print(train_cp.y)
            # quit()

            # ======= pre train


            # ======= train
            print('==== train cyclic peptide data')
            # classification_rms = dc.metrics.Metric(dc.metrics.roc_auc_score, mode=mode)
            model = trainer_regression(model, n_epoch=n_epoch, patience=patience, train_data=train_cp,
                            valid_data=valid_cp, metrics=[rms], transformers=[],
                            text=f'Training permeability with seed {actual_seed}')

            print('Confirm valid loss', model.evaluate(valid_cp, [rms])['rms_score'])
            test_pred = model.predict(test_cp)
            print('RMSE for test data', mean_squared_error(model.predict(test_cp), test_cp.y, squared=False))
            print('MAE for test data', mean_absolute_error(test_pred, test_cp.y))
            # print('test pred rmse:', np.mean((model.predict(test_cp) - test_cp.y)**2)**0.5)
            df[f'Pred_{actual_seed}'] = test_pred
            df[f'True_{actual_seed}'] = test_cp.y
            model.load_from_dir()
        # print(df.keys())
        df.to_csv(f'./CSV/Predictions/TVT_Scaffold_Split/Trained_on_6&7&10/{m_name}.csv', index=False)


if __name__ == '__main__':
    n_epoch = 20000
    batch_size = 64
    patience = 200
    task = ["Normalized_PAMPA"]
    # task = "Binary"
    mode = "regression"
    # mode = "classification"

    # train_list = [f'./CSV/mol_length_6_blk{i}.csv' for i in range(1, 9)]
    train_list = [f'./CSV/Data/mol_length_{i}_train.csv' for i in [6, 7, 10]]
    # train_list += [f'./CSV/mol_length_{i}.csv' for i in [2, 3, 4, 5, 8, 9, 11, 12, 13, 14, 15]]
    valid_list = [f'./CSV/Data/mol_length_{i}_valid.csv' for i in [6, 7, 10]]
    test_list = [f'./CSV/Data/mol_length_{i}_test.csv' for i in [6, 7, 10]]
    print(train_list)
    print(valid_list)
    print(test_list)

    seed_list_ = [123 * i ** 2 for i in range(1)]
    model_dir = "../SavedModel/DeepChemPermeability/TVT_split/"

    model_list = ['DMPNN', 'GCN', 'GAT', 'MPNN', 'PAGTN', 'AttentiveFP']
    # main(model_list[:1] + model_list[2:3], model_dir, seed_list_)
    main(['DMPNN'], model_dir, seed_list_)
