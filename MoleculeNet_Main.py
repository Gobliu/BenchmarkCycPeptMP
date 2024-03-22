import os
import sys
import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_squared_error

import deepchem as dc
import deepchem.models.losses as losses
import pandas as pd
import torch.nn as nn

from Utils import manual_seed


def data_processor(loader):
    df = pd.read_csv('CycPep.csv')
    grouped = df.groupby('Split')
    for group_name, group_df in grouped:
        group_df.to_csv(f'group_{group_name}.csv', index=False)
    print("Groups saved to CSV files.")
    train = loader.create_dataset('group_train.csv')
    valid = loader.create_dataset('group_valid.csv')
    test = loader.create_dataset('group_test.csv')
    mean = np.mean(train.y)
    std = np.std(train.y)
    transformer = dc.trans.NormalizationTransformer(transform_y=True, dataset=train)
    train = transformer.transform(train)
    valid = transformer.transform(valid)
    test = transformer.transform(test)
    print(f'Got {len(train.y)} samples for train, {len(valid.y)} for valid, and {len(test.y)} for test')
    return {'train': train, 'valid': valid, 'test': test, 'mean': mean, 'std': std, 'transformer': transformer}


def main(feat, model, m_name, seed):
    loader = dc.data.CSVLoader(tasks=['PAMPA'],
                               feature_field="smiles",
                               featurizer=feat)

    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer=featurizer, splitter='random')
    train_sol, valid_sol, test_sol = datasets

    data = data_processor(loader)
    train_cp, valid_cp, test_cp = data['train'], data['valid'], data['test']
    df = pd.read_csv('group_test.csv')

    for _ in range(10):
        manual_seed(seed)
        # ======= pre train
        print('==== train solubility data')

        for i in range(n_epoch):
            loss = model.fit(train_sol, nb_epoch=1)
            print('train delaney, epoch:', i, 'loss', loss)

        # ======= train
        print('==== train cyclic peptide data')
        transformer = data['transformer']
        rms = dc.metrics.Metric(dc.metrics.score_function.rms_score)
        current_loss = float('inf')
        current_patient = 0
        for i in range(n_epoch):
            loss = model.fit(train_cp, nb_epoch=1)
            valid_loss = model.evaluate(valid_cp, [rms], transformers=[transformer])['rms_score']
            print('train permeability epoch:', i, 'loss', loss, 'valid loss', valid_loss)
            if valid_loss < current_loss:
                current_loss = valid_loss
                current_patient = 0
                best_model = deepcopy(model)
            else:
                current_patient += 1

            if current_patient > patience:
                print(f"val_loss {current_loss} did not decrease for {current_patient} epochs consequently.")
                break

        print('Confirm valid loss', best_model.evaluate(valid_cp, [rms], transformers=[transformer])['rms_score'])
        test_pred = best_model.predict(test_cp, transformers=[transformer])
        print('RMSE for test data', mean_squared_error(best_model.predict(test_cp, transformers=[transformer]),
                                                       transformer.untransform(test_cp.y), squared=False))
        print('test pred rmse:', np.mean((best_model.predict(test_cp) - test_cp.y)**2)**0.5)
        df[f'Pred_{seed}'] = test_pred
        df[f'True_{seed}'] = transformer.untransform(test_cp.y)
        seed *= 3

    df.to_csv(f'{m_name}.csv', index=False)


if __name__ == '__main__':
    n_epoch = 1000
    batch_size = 64
    patience = 80

    featurizer = dc.feat.MolGraphConvFeaturizer()
    model_name = 'GCN'
    net = dc.models.GCNModel(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)
    main(feat=featurizer, model=net, m_name=model_name, seed=123)

    model_name = 'GAT'
    net = dc.models.GATModel(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)
    main(feat=featurizer, model=net, m_name=model_name, seed=123)

    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    model_name = 'AttentiveFP'
    net = dc.models.AttentiveFPModel(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)
    main(feat=featurizer, model=net, m_name=model_name, seed=123)

    model_name = 'MPNN'
    net = dc.models.torch_models.MPNNModel(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)
    main(feat=featurizer, model=net, m_name=model_name, seed=123)

    featurizer = dc.feat.PagtnMolGraphFeaturizer()
    model_name = 'PAGTN'
    net = dc.models.PagtnModel(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)
    main(feat=featurizer, model=net, m_name=model_name, seed=123)

    # y_true = np.array([3, -0.5, 2, 7])
    # y_pred = np.array([2.5, 0.0, 2, 8])
    # print(mean_squared_error(y_true, y_pred))
    # print(np.mean((y_true - y_pred)**2))
    # print((y_true - y_pred) ** 2)
