import os
import sys
import numpy as np
# import random
from copy import deepcopy
import yaml

import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

import deepchem as dc
import deepchem.models.losses as losses
import pandas as pd
import torch.nn as nn

from CustomizedDateLoader import *
from ModelFeatureGenerator import generate_model_feature
from ModelTrainer import model_trainer
from Utils import set_seed


def main(m_names):
    for model_name in m_names:
        print(f'==== training {model_name} model ====')

        rms, rms_score = rms_dict[args['mode']]
        tasks = task_dict[args['mode']]
        for split_seed in range(1, 11):
            set_seed(123 * split_seed ** 2)
            feat, model = generate_model_feature(model_name, len(tasks), args)
            # model.loss = dc.models.losses.SoftmaxCrossEntropy
            # print(model.loss)
            print('==== pre train ====')
            if args['mode'] == 'regression':
                _, datasets_pre_train, transformers_pre_train = dc.molnet.load_delaney(
                    featurizer=feat, splitter='random'
                )
            elif args['mode'] == 'classification' or args['mode'] == 'soft':
                _, datasets_pre_train, transformers_pre_train = dc.molnet.load_bbbp(
                    featurizer=feat, splitter='random'
                )

            train_pre_train, valid_pre_train, test_pre_train = datasets_pre_train

            if len(tasks) > 1:
                train_pre_train = convert_multitask(train_pre_train, len(tasks))
                valid_pre_train = convert_multitask(valid_pre_train, len(tasks))
                test_pre_train = convert_multitask(test_pre_train, len(tasks))

            model = model_trainer(
                model, f"{args['model_dir']}/{args['split']}/{args['mode']}/{model_name}",
                train_pre_train, valid_pre_train,
                metrics=[rms], score_name=rms_score, transformers=transformers_pre_train,
                text=f'Pre-Training with seed {split_seed}', args=args
            )

            print('==== train cyclic peptide data ====')
            loader = dc.data.CSVLoader(
                tasks=tasks, feature_field="SMILES",
                id_field="Original_Name_in_Source_Literature", featurizer=feat
            )

            if args['split'] == 'scaffold':
                assert len(csv_list) == 3, 'Expect a list of [train_list, valid_list, test_list]'
                data = data_loader_separate(csv_list, loader)
            elif args['split'] == 'random':
                data = data_loader_all_in_one(csv_list, split_seed, loader)

            train_cp, valid_cp, test_cp, test_df = data['train'], data['valid'], data['test'], data['test_df']

            if args['mode'] == 'classification' or args['mode'] == 'soft':
                train_cp, valid_cp, test_cp = adjust_label_weights(train_cp, valid_cp, test_cp, [1.])

            if args['mode'] == 'soft':
                valid_cp = soft_label2hard(valid_cp)
                test_cp = soft_label2hard(test_cp)

            # classification_rms = dc.metrics.Metric(dc.metrics.roc_auc_score, mode=mode)
            model = model_trainer(
                model, f"{args['model_dir']}/{args['split']}/{args['mode']}/{model_name}", train_cp, valid_cp,
                metrics=[rms], score_name=rms_score, transformers=[],
                text=f'Training permeability with seed {split_seed}', args=args
            )

            print(f"{args['model_dir']}/{args['split']}/{args['mode']}/{model_name}/checkpoint1.pt",
                  f"{args['model_dir']}/{args['split']}/{args['mode']}/{model_name}/checkpoint_seed{split_seed}.pt")
            os.rename(f"{args['model_dir']}/{args['split']}/{args['mode']}/{model_name}/checkpoint1.pt",
                      f"{args['model_dir']}/{args['split']}/{args['mode']}/{model_name}/checkpoint_seed{split_seed}.pt")
            print('Confirm valid loss', model.evaluate(valid_cp, [rms])[rms_score])

            if args['mode'] == 'regression':
                test_df[f'Pred_{split_seed}'] = model.predict(test_cp)
            elif args['mode'] == 'classification' or args['mode'] == 'soft':
                test_df[f'Pred_{split_seed}'] = model.predict(test_cp)[:, 1]
            test_df[f'True_{split_seed}'] = test_cp.y

            test_csv_path = f"{args['csv_dir']}/{args['split']}/{args['mode']}/{model_name}_seed{split_seed}.csv"
            print('Saving csv of test data to', test_csv_path)
            test_df.to_csv(test_csv_path, index=False)


if __name__ == '__main__':
    yaml_config_path = "Config.yaml"
    with open(yaml_config_path, "r") as f:
        args = yaml.load(f, Loader=yaml.Loader)

    rms_dict = {'classification': [dc.metrics.Metric(dc.metrics.prc_auc_score), 'prc_auc_score'],
                'regression': [dc.metrics.Metric(dc.metrics.score_function.rms_score), 'rms_score'],
                'soft': [dc.metrics.Metric(dc.metrics.prc_auc_score), 'prc_auc_score']}

    task_dict = {'classification': ['Binary'],
                 'regression': ['Normalized_PAMPA'],
                 'soft': ['Soft_Label']}

    if args['split'] == 'scaffold':
        mol_length_list = [6, 7, 10]
        csv_list = [[f'./CSV/Data/mol_length_{i}_train.csv' for i in mol_length_list],
                    [f'./CSV/Data/mol_length_{i}_valid.csv' for i in mol_length_list],
                    [f'./CSV/Data/mol_length_{i}_test.csv' for i in mol_length_list]]
    elif args['split'] == 'random':
        csv_list = ["./CSV/Data/Random_Split.csv"]

    print('Working on csv list:', csv_list)
    model_list = ['DMPNN', 'GCN', 'GAT', 'MPNN', 'PAGTN', 'AttentiveFP']
    main(model_list)
