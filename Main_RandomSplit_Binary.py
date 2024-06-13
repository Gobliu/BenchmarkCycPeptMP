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
    bce = dc.metrics.Metric(dc.metrics.score_function.prc_auc_score)
    ip_df = pd.read_csv(ip_path)

    for split_seed in range(10):
        manual_seed(123 * split_seed ** 2)
        feat, model = generate_model_feature(m_name, op_dir, batch_size=batch_size, mode=mode)

        # ======= pre train
        tasks, datasets_pre_train, transformers_pre_train = dc.molnet.load_bbbp(featurizer=feat, splitter='random')
        train_pre_train, valid_pre_train, test_pre_train = datasets_pre_train
        print(test_pre_train)
        print('==== train pre training data')

        model = trainer_classification(model, n_epoch=n_epoch // 5, patience=patience, train_data=train_pre_train,
                                       valid_data=valid_pre_train, metrics=[bce], transformers=transformers_pre_train,
                                       text=f'Training solubility with split seed {split_seed}')

        print('Confirm valid loss for pre train',
              model.evaluate(valid_pre_train, [bce], transformers_pre_train)['prc_auc_score'])

        # ======= train
        loader = dc.data.CSVLoader(tasks=[task],
                                   feature_field="SMILES",
                                   id_field="Original_Name_in_Source_Literature",
                                   featurizer=feat)

        data = data_loader(ip_df, split_seed, loader)
        train_cp, valid_cp, test_cp = data['train'], data['valid'], data['test']
        df = pd.read_csv('temp_test.csv')
        print('==== train cyclic peptide data')
        model = trainer_classification(model, n_epoch=n_epoch, patience=patience, train_data=train_cp,
                                       valid_data=valid_cp, metrics=[bce], transformers=[],
                                       text=f'Training permeability with split seed {split_seed}')

        print('Confirm train loss', model.evaluate(train_cp, [bce])['prc_auc_score'])
        print('Confirm valid loss', model.evaluate(valid_cp, [bce])['prc_auc_score'])
        # print('train data true', train_cp.y[:, 0])
        # print('train data pred', model.predict(train_cp)[:, 0])
        print('BCE for Train', log_loss(train_cp.y, model.predict(train_cp)[:, 1]))
        precision, recall, _ = precision_recall_curve(train_cp.y, model.predict(train_cp)[:, 1])
        print('ROC AUC for Train', auc(recall, precision))
        # test_pred = model.predict(test_cp)
        # print(test_cp.y.shape, test_cp.y)
        # print(type(test_pred), test_pred.shape, test_pred)
        # print('RMSE for test data', mean_squared_error(model.predict(test_cp), test_cp.y, squared=False))
        # print('MAE for test data', mean_absolute_error(test_pred, test_cp.y))
        # print('test pred rmse:', np.mean((model.predict(test_cp) - test_cp.y)**2)**0.5)
        # print('BCE for test', log_loss(test_cp.y, test_pred))
        df[f'Pred_{split_seed}'] = model.predict(test_cp)[:, 1]
        df[f'True_{split_seed}'] = test_cp.y
        precision, recall, _ = precision_recall_curve(test_cp.y, model.predict(test_cp)[:, 1])
        print('ROC AUC for Train', auc(recall, precision))
        model.save_checkpoint()

        df.to_csv(f'./CSV/Predictions/TVT_Random_Split/Binary/{m_name}_SplitSeed{split_seed}.csv', index=False)


if __name__ == '__main__':
    n_epoch = 2
    batch_size = 64
    patience = 200
    task = "Binary"
    mode = "classification"
    model_dir = "../SavedModel/DeepChemPermeability/TVT_split/"

    # model_list = ['DMPNN', 'GCN', 'GAT', 'MPNN', 'PAGTN', 'AttentiveFP']
    csv_path = "./CSV/Data/Random_Split.csv"
    main(csv_path, model_dir, m_name="AttentiveFP")
