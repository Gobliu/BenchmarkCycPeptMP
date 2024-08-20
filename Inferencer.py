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

        feat, model = generate_model_feature(model_name, len(tasks), args)

        loader = dc.data.CSVLoader(
            tasks=tasks, feature_field="SMILES",
            id_field="Original_Name_in_Source_Literature", featurizer=feat
        )

        for csv in csv_list:
            test_cp = loader.create_dataset(csv)
            test_df = pd.read_csv(csv)
            csv_name = csv.split('/')[-1]
            print(csv_name)
            for split_seed in range(1, 11):
                # data = data_loader_all_in_one(csv_list, split_seed, loader)
                # train_cp, valid_cp, test_cp, test_df = data['train'], data['valid'], data['test'], data['test_df']
                wp = f"{args['model_dir']}/{args['split']}/{args['mode']}/{model_name}/checkpoint_seed{split_seed}.pt"
                model.restore(wp)
                print(model.loss)
                if args['mode'] == 'regression':
                    test_df[f'Pred_{split_seed}'] = model.predict(test_cp)
                elif args['mode'] == 'classification' or args['mode'] == 'soft':
                    test_df[f'Pred_{split_seed}'] = model.predict(test_cp)[:, 1]
                    loss = model.evaluate(test_cp, [rms], [])[rms_score]
                    # print(loss)
                    # print(model.predict(test_cp), test_cp.y)
                    # torch_ce_loss = nn.CrossEntropyLoss()
                    # print(torch_ce_loss(torch.from_numpy(model.predict(test_cp)), torch.tensor([[1, 0]]).float()))
                    # quit()
                test_df[f'True_{split_seed}'] = test_cp.y

                test_csv_path = f"{args['csv_dir']}/{args['split']}/{args['mode']}/{model_name}_{csv_name}"
            print('Saving csv of test data to', test_csv_path)
            # test_df.to_csv(test_csv_path, index=False)


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

    csv_list = [f'./CSV/Data/mol_length_{i}.csv' for i in [8, 9]]
    # csv_list = ["./CSV/Data/Random_Split.csv"]

    print('Working on csv list:', csv_list)
    model_list = ['DMPNN', 'GCN', 'GAT', 'MPNN', 'PAGTN', 'AttentiveFP']
    main(model_list[:1])
