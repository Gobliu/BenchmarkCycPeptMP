import os
import sys

import numpy as np
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


def rms_loss(pred, target, weight):
    print(pred.dtype)
    print(pred.shape, target.shape, weight.shape)
    error = np.sum((pred - target)**2)
    return error


def main(seed):
    model_name = 'DMPNN'
    featurizer = dc.feat.DMPNNFeaturizer()
    loader = dc.data.CSVLoader(tasks=['PAMPA'],
                               feature_field="smiles",
                               featurizer=featurizer)
    model = dc.models.DMPNNModel(n_tasks=1, mode='regression', model_dir=model_name, batch_size=batch_size)

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
            # print('train delaney, epoch:', i, 'loss', loss)

        # ======= train
        print('==== train cyclic peptide data')
        # mean, std = data['mean'], data['std']
        transformer = data['transformer']
        rms = dc.metrics.Metric(dc.metrics.score_function.rms_score)
        current_loss = float('inf')
        current_patient = 0
        checkpoint_interval = int(np.ceil(len(train_cp.y) / batch_size))
        # print(checkpoint_interval)
        for i in range(n_epoch):
            loss = model.fit(train_cp, nb_epoch=1, max_checkpoints_to_keep=patience,
                             checkpoint_interval=checkpoint_interval*4)
            valid_loss = model.evaluate(valid_cp, [rms], transformers=[transformer])['rms_score']
            # print('train permeability epoch:', i, 'loss', loss, 'valid loss', valid_loss)
            if valid_loss < current_loss:
                current_loss = valid_loss
                current_patient = 0
            else:
                current_patient += 1

            if current_patient > patience:
                print(f"val_loss {valid_loss} did not decrease for {current_patient} epochs consequently.")
                break

        valid_losses1 = []
        valid_losses2 = []
        for i in range(patience):
            model.restore(checkpoint=f"{model_name}/checkpoint{i+1}.pt")
            # print('checkpoint', i+1, model.evaluate(valid_cp, [rms]))
            valid_losses1.append(model.evaluate(valid_cp, [rms], transformers=[transformer])['rms_score'])
            valid_losses2.append(mean_squared_error(model.predict(valid_cp), valid_cp.y))

        pos = valid_losses1.index(min(valid_losses1))
        model.restore(checkpoint=f"{model_name}/checkpoint{pos+1}.pt")
        print(min(valid_losses1), model.evaluate(valid_cp, [rms], transformers=[transformer])['rms_score'])
        print(min(valid_losses2), valid_losses2.index(min(valid_losses2)), pos)
        test_pred = model.predict(test_cp, transformers=[transformer])
        print('test pred rms:', (np.mean(model.predict(test_cp) - test_cp.y)**2)**0.5)

        df[f'Pred{seed}'] = test_pred
        seed *= 5

    df.to_csv('test_dmpnn.csv', index=False)


if __name__ == '__main__':
    n_epoch = 1000
    batch_size = 64
    patience = 80
    main(seed=123)
