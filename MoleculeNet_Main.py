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


def data_processor(train_list, valid_list, test_list, loader):
    train = loader.create_dataset(train_list)
    valid = loader.create_dataset(valid_list)
    test = loader.create_dataset(test_list)
    mean = np.mean(train.y)
    std = np.std(train.y)
    transformer = dc.trans.NormalizationTransformer(transform_y=True, dataset=train)
    train = transformer.transform(train)
    valid = transformer.transform(valid)
    test = transformer.transform(test)
    print(f'Got {len(train.y)} samples for train, {len(valid.y)} for valid, and {len(test.y)} for test')
    return {'train': train, 'valid': valid, 'test': test, 'mean': mean, 'std': std, 'transformer': transformer}


def trainer(model, n_epoch, patience, train_data, valid_data, metrics, transformers, text):
    current_loss = float('inf')
    current_patient = 0
    best_model = deepcopy(model)
    for i in range(n_epoch):
        loss = model.fit(train_data, nb_epoch=1, checkpoint_interval=0)
        valid_loss = model.evaluate(valid_data, metrics, transformers)['rms_score']
        print(text, 'epoch', i, 'loss', loss, 'valid loss', valid_loss)
        if valid_loss < current_loss:
            current_loss = valid_loss
            current_patient = 0
            best_model = deepcopy(model)
        else:
            current_patient += 1

        if current_patient > patience:
            print(f"val_loss {current_loss} did not decrease for {current_patient} epochs consequently.")
            break
    return best_model


def main(feat, model, m_name, seed):
    loader = dc.data.CSVLoader(tasks=['PAMPA'],
                               feature_field="SMILES",
                               id_field="Original_Name_in_Source_Literature",
                               featurizer=feat)

    tasks, datasets, transformers_sol = dc.molnet.load_delaney(featurizer=featurizer, splitter='random')
    train_sol, valid_sol, test_sol = datasets
    rms = dc.metrics.Metric(dc.metrics.score_function.rms_score)

    data = data_processor(train_list, valid_list, test_list, loader)
    train_cp, valid_cp, test_cp = data['train'], data['valid'], data['test']

    df_list = []
    for f in test_list:
        df_list.append(pd.read_csv(f))
    df = pd.concat(df_list, ignore_index=True)

    for actual_seed in seed_list:
        manual_seed(actual_seed)
        # ======= pre train
        print('==== train solubility data')

        model = trainer(model, n_epoch=n_epoch, patience=patience, train_data=train_sol,
                        valid_data=valid_sol, metrics=[rms], transformers=transformers_sol,
                        text=f'Training solubility with seed {actual_seed}')

        # ======= train
        print('==== train cyclic peptide data')
        transformer_cp = data['transformer']
        model = trainer(model, n_epoch=n_epoch, patience=patience, train_data=train_cp,
                        valid_data=valid_cp, metrics=[rms], transformers=[transformer_cp],
                        text=f'Training permeability with seed {actual_seed}')

        print('Confirm valid loss', model.evaluate(valid_cp, [rms], transformers=[transformer_cp])['rms_score'])
        test_pred = model.predict(test_cp, transformers=[transformer_cp])
        print('RMSE for test data', mean_squared_error(model.predict(test_cp, transformers=[transformer_cp]),
                                                       transformer_cp.untransform(test_cp.y), squared=False))
        print('test pred rmse:', np.mean((model.predict(test_cp) - test_cp.y)**2)**0.5)
        df[f'Pred_{seed}'] = test_pred
        df[f'True_{seed}'] = transformer_cp.untransform(test_cp.y)
        model.save_checkpoint()
        model = dc.models.DMPNNModel(n_tasks=1, mode='regression', model_dir=f"{model_dir}{model_name}",
                                     batch_size=batch_size)
        # model.restore()
        # test_pred = model.predict(test_cp, transformers=[transformer_cp])
        print('Confirm valid loss', model.evaluate(valid_cp, [rms], transformers=[transformer_cp])['rms_score'])
        print('RMSE for test data', mean_squared_error(model.predict(test_cp, transformers=[transformer_cp]),
                                                       transformer_cp.untransform(test_cp.y), squared=False))
        print('test pred rmse:', np.mean((model.predict(test_cp) - test_cp.y)**2)**0.5)
        # df[f'Pred_{seed}'] = test_pred
        # df[f'True_{seed}'] = transformer_cp.untransform(test_cp.y)

    df.to_csv(f'./CSV/{m_name}.csv', index=False)


if __name__ == '__main__':
    n_epoch = 20000
    batch_size = 64
    patience = 200

    train_list = ['./CSV/mol_length_6_train.csv']
    valid_list = ['./CSV/mol_length_6_valid.csv']
    test_list = ['./CSV/mol_length_6_test.csv']

    seed_list = [123*3**i for i in range(10)]
    model_dir = "../SavedModel/DeepChemPermeability/"

    # featurizer = dc.feat.MolGraphConvFeaturizer()
    # model_name = 'GCN'
    # net = dc.models.GCNModel(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)
    # main(feat=featurizer, model=net, m_name=model_name, seed=123)
    #
    # model_name = 'GAT'
    # net = dc.models.GATModel(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)
    # main(feat=featurizer, model=net, m_name=model_name, seed=123)

    # featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    # model_name = 'AttentiveFP'
    # net = dc.models.AttentiveFPModel(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)
    # main(feat=featurizer, model=net, m_name=model_name, seed=123)
    #
    # model_name = 'MPNN'
    # net = dc.models.torch_models.MPNNModel(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)
    # main(feat=featurizer, model=net, m_name=model_name, seed=123)
    #
    # featurizer = dc.feat.PagtnMolGraphFeaturizer()
    # model_name = 'PAGTN'
    # net = dc.models.PagtnModel(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)
    # main(feat=featurizer, model=net, m_name=model_name, seed=123)

    featurizer = dc.feat.DMPNNFeaturizer()
    model_name = 'DMPNN'
    net = dc.models.DMPNNModel(n_tasks=1, mode='regression', model_dir=f"{model_dir}{model_name}",
                               batch_size=batch_size)

    main(feat=featurizer, model=net, m_name=model_name, seed=123)
    # y_true = np.array([3, -0.5, 2, 7])
    # y_pred = np.array([2.5, 0.0, 2, 8])
    # print(mean_squared_error(y_true, y_pred))
    # print(np.mean((y_true - y_pred)**2))
    # print((y_true - y_pred) ** 2)
