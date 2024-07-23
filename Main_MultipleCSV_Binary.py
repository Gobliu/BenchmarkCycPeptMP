import os
import sys
import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, precision_recall_curve, auc

import deepchem as dc
import deepchem.models.losses as losses
import pandas as pd
import torch.nn as nn

from Utils import manual_seed
from ModelFeatureGenerator import generate_model_feature
from Trainer import trainer_classification


def data_loader(train_list, valid_list, test_list, loader):
    train = loader.create_dataset(train_list)
    valid = loader.create_dataset(valid_list)
    test = loader.create_dataset(test_list)
    print(f'Got {len(train.y)} samples for train, {len(valid.y)} for valid, and {len(test.y)} for test')
    return {'train': train, 'valid': valid, 'test': test}


def main(op_dir, m_name):
    print(f'==== training {m_name} model')
    bce = dc.metrics.Metric(dc.metrics.score_function.prc_auc_score)
    df_list = []
    for f in test_list:
        df_list.append(pd.read_csv(f))
    df = pd.concat(df_list, ignore_index=True)

    for s_seed in range(10):
        actual_seed = 123 * s_seed ** 2
        manual_seed(actual_seed)
        feat, model = generate_model_feature(m_name, op_dir, batch_size=batch_size, mode=mode)

        # ======= pre train
        tasks, datasets_pre_train, transformers_pre_train = dc.molnet.load_bbbp(featurizer=feat, splitter='random')
        train_pre_train, valid_pre_train, test_pre_train = datasets_pre_train

        # ======= pre train
        print('==== train solubility data')

        model = trainer_classification(
            model, n_epoch=n_epoch // 5, patience=patience, train_data=train_pre_train,
            valid_data=valid_pre_train, metrics=[bce], transformers=transformers_pre_train,
            text=f'Training solubility with seed {s_seed}')

        # ======= train

        loader = dc.data.CSVLoader(tasks=[task],
                                   feature_field="SMILES",
                                   id_field="Original_Name_in_Source_Literature",
                                   featurizer=feat)

        data = data_loader(train_list, valid_list, test_list, loader)
        train_cp, valid_cp, test_cp = data['train'], data['valid'], data['test']
        print('==== train cyclic peptide data')
        # classification_rms = dc.metrics.Metric(dc.metrics.roc_auc_score, mode=mode)
        model = trainer_classification(
            model, n_epoch=n_epoch, patience=patience, train_data=train_cp,
            valid_data=valid_cp, metrics=[bce], transformers=[],
            text=f'Training permeability with seed {s_seed}')

        print('Confirm train loss', model.evaluate(train_cp, [bce])['prc_auc_score'])
        print('Confirm valid loss', model.evaluate(valid_cp, [bce])['prc_auc_score'])
        print('BCE for Train', log_loss(train_cp.y, model.predict(train_cp)[:, 1]))
        precision, recall, _ = precision_recall_curve(train_cp.y, model.predict(train_cp)[:, 1])
        print('ROC AUC for Train', auc(recall, precision))
        df[f'Pred_{s_seed}'] = model.predict(test_cp)[:, 1]
        df[f'True_{s_seed}'] = test_cp.y
        precision, recall, _ = precision_recall_curve(test_cp.y, model.predict(test_cp)[:, 1])
        print('ROC AUC for Train', auc(recall, precision))
        model.save_checkpoint()
        df.to_csv(f'./CSV/Predictions/TVT_Scaffold_Split/Trained_on_all/Binary/{m_name}_ModelSeed{actual_seed}.csv', index=False)


if __name__ == '__main__':
    n_epoch = 20000
    batch_size = 64
    patience = 2001
    # task = "Normalized_PAMPA"
    task = "Binary"
    # mode = "regression"
    mode = "classification"

    train_list = [f'./CSV/Data/mol_length_{i}_train.csv' for i in [6, 7, 10]]
    train_list += [f'./CSV/Data/mol_length_{i}_train.csv' for i in [2, 3, 4, 5, 8, 9, 11, 12, 13, 14, 15]]
    valid_list = [f'./CSV/Data/mol_length_{i}_valid.csv' for i in [6, 7, 10]]
    test_list = [f'./CSV/Data/mol_length_{i}_test.csv' for i in [6, 7, 10]]
    print(train_list)
    print(valid_list)
    print(test_list)

    seed_list_ = [123 * i ** 2 for i in range(2)]
    model_dir = "../SavedModel/DeepChemPermeability/TVT_split/"

    model_list = ['DMPNN', 'GCN', 'GAT', 'MPNN', 'PAGTN', 'AttentiveFP']
    for m in model_list[4:]:
        main(model_dir, m)
