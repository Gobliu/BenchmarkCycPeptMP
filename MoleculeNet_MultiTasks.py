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


def split_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=[])
    grouped = df.groupby('Block')
    for group_name, group_df in grouped:
        group_df.to_csv(f'Block_{group_name}.csv', index=False)
    print("Groups saved to CSV files.")


def data_loader(loader, train_list, valid_list, test_list):
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
        loss = model.fit(train_data, nb_epoch=1)
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


def main(feat, model, m_name, seed_factor):
    loader = dc.data.CSVLoader(tasks=['PAMPA', 'MolLogP'],
                               feature_field="SMILES",
                               id_field="Comp",
                               featurizer=feat)

    tasks_qm9, datasets_qm9, transformers_qm9 = dc.molnet.load_qm9(featurizer=featurizer, splitter='random')
    train_qm9, valid_qm9, test_qm9 = datasets_qm9
    tasks_qm9 = tasks_qm9[:2]

    rms = dc.metrics.Metric(dc.metrics.score_function.rms_score)

    blk_size = 10
    n_seed = 10
    # for idx in range(6, blk_size):
    for idx in range(6):
        test_idx = idx
        valid_idx = (idx + 1) % blk_size
        idx_list = [x for x in range(blk_size) if x not in [valid_idx, test_idx]]
        train_list = [f"Block_{i}.csv" for i in idx_list]
        valid_list = [f"Block_{valid_idx}.csv"]
        test_list = [f"Block_{test_idx}.csv"]
        data = data_loader(loader, train_list, valid_list, test_list)
        train_cp, valid_cp, test_cp = data['train'], data['valid'], data['test']
        df = pd.read_csv(f"Block_{test_idx}.csv")

        for repeat in range(n_seed):
            actual_seed = seed_factor * 3 ** repeat
            manual_seed(actual_seed)
            # ======= pre train
            print('==== train solubility data')
            # Create the weighted multitask objective

            # Compile the model with the weighted multitask objective
            model.compile(metrics=[dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")], loss=loss,
                          optimizer='adam')

            model = trainer(model, n_epoch=n_epoch_, patience=patience_, train_data=train_qm9,
                            valid_data=valid_qm9, metrics=[rms], transformers=transformers_qm9,
                            text=f'Training solubility with seed {actual_seed}')

            # ======= train
            print('==== train cyclic peptide data')
            transformer_cp = data['transformer']
            model = trainer(model, n_epoch=n_epoch_, patience=patience_, train_data=train_cp,
                            valid_data=valid_cp, metrics=[rms], transformers=[transformer_cp],
                            text=f'Training solubility with seed {actual_seed}')

            print('Confirm valid loss', model.evaluate(valid_cp, [rms], transformers=[transformer_cp])['rms_score'])
            test_pred = model.predict(test_cp, transformers=[transformer_cp])
            print('RMSE for test data', mean_squared_error(model.predict(test_cp, transformers=[transformer_cp]),
                                                           transformer_cp.untransform(test_cp.y), squared=False))
            # TODO ===== untransformed result below, why it is smaller??
            print('test pred rmse:', np.mean((model.predict(test_cp) - test_cp.y)**2)**0.5)
            df[f'Pred_{actual_seed}'] = test_pred
            df[f'True_{actual_seed}'] = transformer_cp.untransform(test_cp.y)

        df.to_csv(f'new_{m_name}_block{idx}.csv', index=False)


if __name__ == '__main__':
    n_epoch_ = 1000
    batch_size = 64
    patience_ = 80

    split_csv('2015_Wan_2016_Fur_2018_Nay_2022_Tae-10Blocks.csv')

    featurizer = dc.feat.MolGraphConvFeaturizer()
    # model_name = 'GCN'
    # net = dc.models.GCNModel(n_tasks=2, mode='regression', model_dir=model_name, batch_size=batch_size)
    # main(feat=featurizer, model=net, m_name=model_name, seed=123)
    #
    model_name = 'GAT'
    net = dc.models.GATModel(n_tasks=2, mode='regression', model_dir=model_name, batch_size=batch_size)
    main(feat=featurizer, model=net, m_name=model_name, seed_factor=123)

    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    # model_name = 'AttentiveFP'
    # net = dc.models.AttentiveFPModel(n_tasks=2, mode='regression', model_dir=model_name, batch_size=batch_size)
    # main(feat=featurizer, model=net, m_name=model_name, seed_factor=123)

    model_name = 'MPNN'
    net = dc.models.torch_models.MPNNModel(n_tasks=2, mode='regression', model_dir=model_name, batch_size=batch_size)
    main(feat=featurizer, model=net, m_name=model_name, seed_factor=123)

    featurizer = dc.feat.PagtnMolGraphFeaturizer()
    model_name = 'PAGTN'
    net = dc.models.PagtnModel(n_tasks=2, mode='regression', model_dir=model_name, batch_size=batch_size)
    main(feat=featurizer, model=net, m_name=model_name, seed_factor=123)
