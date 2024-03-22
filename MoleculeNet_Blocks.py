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


def main(feat, model, m_name, seed_list):
    loader = dc.data.CSVLoader(tasks=['PAMPA'],
                               feature_field="SMILES",
                               id_field="Comp",
                               featurizer=feat)

    tasks, datasets_sol, transformers_sol = dc.molnet.load_delaney(featurizer=featurizer, splitter='random')
    train_sol, valid_sol, test_sol = datasets_sol

    rms = dc.metrics.Metric(dc.metrics.score_function.rms_score)

    blk_size = 10
    # n_seed = 10
    for idx in range(1, blk_size):
        test_idx = idx
        valid_idx = (idx + 1) % blk_size
        idx_list = [x for x in range(blk_size) if x not in [valid_idx, test_idx]]
        train_list = [f"Block_{i}.csv" for i in idx_list]
        valid_list = [f"Block_{valid_idx}.csv"]
        test_list = [f"Block_{test_idx}.csv"]
        data = data_loader(loader, train_list, valid_list, test_list)
        train_cp, valid_cp, test_cp = data['train'], data['valid'], data['test']
        df = pd.read_csv(f"Block_{test_idx}.csv")

        for actual_seed in seed_list:
            manual_seed(actual_seed)
            # ======= pre train
            print('==== train solubility data')

            # model = trainer(model, n_epoch=n_epoch_, patience=patience_, train_data=train_sol,
            #                 valid_data=valid_sol, metrics=[rms], transformers=transformers_sol,
            #                 text=f'Training solubility with seed {actual_seed}')

            # ======= train
            print('==== train cyclic peptide data')
            transformer_cp = data['transformer']
            # model = trainer(model, n_epoch=n_epoch_, patience=patience_, train_data=train_cp,
            #                 valid_data=valid_cp, metrics=[rms], transformers=[transformer_cp],
            #                 text=f'Training solubility with seed {actual_seed}')

            model.reload()
            print('Confirm valid loss', model.evaluate(valid_cp, [rms], transformers=[transformer_cp])['rms_score'])
            test_pred = model.predict(test_cp, transformers=[transformer_cp])
            print('RMSE for test data', mean_squared_error(model.predict(test_cp, transformers=[transformer_cp]),
                                                           transformer_cp.untransform(test_cp.y), squared=False))
            # print('RMSE for test data', np.mean((model.predict(test_cp, transformers=[transformer_cp]) -
            #                                      transformer_cp.untransform(test_cp.y))**2)**0.5)
            # print(np.mean(test_pred), np.std(test_pred))
            # print(np.mean(model.predict(test_cp)), np.std(model.predict(test_cp)))
            print('test pred rmse:', np.mean((model.predict(test_cp) - test_cp.y)**2)**0.5)

            df[f'Pred_{actual_seed}'] = test_pred
            df[f'True_{actual_seed}'] = transformer_cp.untransform(test_cp.y)

        df.to_csv(f'./CSV/{m_name}_block{idx}.csv', index=False)


if __name__ == '__main__':
    n_epoch_ = 1000
    batch_size = 64
    patience_ = 80

    split_csv('2015_Wan_2016_Fur_2018_Nay_2022_Tae-10Blocks.csv')
    # seed_list = [20122015, 20122016, 20122017]
    seed_list = [123*3**i for i in range(1, 11)]

    # featurizer = dc.feat.MolGraphConvFeaturizer()
    # model_name = 'GCN'
    # net = dc.models.GCNModel(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)
    # main(feat=featurizer, model=net, m_name=model_name, seed_list=seed_list)
    #
    # model_name = 'GAT'
    # net = dc.models.GATModel(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)
    # main(feat=featurizer, model=net, m_name=model_name, seed_list=seed_list)
    #
    # featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    # model_name = 'AttentiveFP'
    # net = dc.models.AttentiveFPModel(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)
    # main(feat=featurizer, model=net, m_name=model_name, seed_list=seed_list)
    #
    # model_name = 'MPNN'
    # net = dc.models.torch_models.MPNNModel(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)
    # main(feat=featurizer, model=net, m_name=model_name, seed_list=seed_list)
    #
    # featurizer = dc.feat.PagtnMolGraphFeaturizer()
    # model_name = 'PAGTN'
    # net = dc.models.PagtnModel(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)
    # main(feat=featurizer, model=net, m_name=model_name, seed_list=seed_list)

    featurizer = dc.feat.DMPNNFeaturizer()
    model_name = 'DMPNN'
    net = dc.models.DMPNNModel(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)
    main(feat=featurizer, model=net, m_name=model_name, seed_list=seed_list)

    # featurizer = dc.feat.ComplexNeighborListFragmentAtomicCoordinates()
    # model_name = 'AtomicConvModel'
    # net = dc.models.AtomicConvModel(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)
    # main(feat=featurizer, model=net, m_name=model_name, seed_list=seed_list)
    #
    # featurizer = dc.feat.SmilesToImage()
    # model_name = 'ChemCeption'
    # net = dc.models.ChemCeption(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)
    # main(feat=featurizer, model=net, m_name=model_name, seed_list=seed_list)
    #
    # featurizer = dc.feat.ConvMolFeaturizer()
    # model_name = 'DAGModel'
    # net = dc.models.DAGModel(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)
    # main(feat=featurizer, model=net, m_name=model_name, seed_list=seed_list)
    #
    # model_name = 'GraphConvModel'
    # net = dc.models.GraphConvModel(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)
    # main(feat=featurizer, model=net, m_name=model_name, seed_list=seed_list)
