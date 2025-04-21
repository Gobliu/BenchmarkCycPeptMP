import os
import sys

import torch.nn
import yaml
from tqdm import tqdm
import pandas as pd

# Dynamically append project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Models import RNN, LSTM, GRU
from DataSet import DataSetSMILES
from ModelHandler import ModelHandler
from Utils import set_seed, get_combined_args


def pytorch_models_inference(m_names):
    torch.cuda.empty_cache()
    kwargs = {'num_workers': 1, 'pin_memory': False} if torch.cuda.is_available() else {}

    dict_path = 'vocab-large.txt'
    x_column = 'SMILES'         # DO NOT convert it into a list
    input_size = 128
    hidden_size = 128
    n_layers = 2
    valid_gap = 2
    sch_step = 100

    args = get_combined_args()

    for model_name in m_names:
        if args.mode == 'regression':
            y_columns = ['Normalized_PAMPA']
            final_act_fun = torch.nn.Identity()
            loss_fun = torch.nn.L1Loss(reduction='none')
        elif args.mode == 'classification':
            y_columns = ['Binary']
            final_act_fun = torch.nn.Sigmoid()
            loss_fun = torch.nn.BCELoss(reduction='none')
        elif args.mode == 'soft':
            y_columns = ['Soft_Label']
            final_act_fun = torch.nn.Sigmoid()
            loss_fun = torch.nn.BCELoss(reduction='none')
        else:
            quit(f"Unknown mode:{args.mode}")

        for split_seed in range(1, 11):
            set_seed(123 * split_seed ** 2)

            test_data = DataSetSMILES(csv_list, dict_path, x_column=x_column, y_columns=y_columns)
            if args.mode == 'classification' or args.mode == 'soft':
                pw, nw = test_data.get_weight()
                test_data.set_weight(pw, nw)

            test_loader = torch.utils.data.DataLoader(test_data, 1, shuffle=False, **kwargs)

            if model_name == 'RNN':
                lr = 0.0001
                net = RNN(input_size, hidden_size, n_layers, len(y_columns),
                          num_embeddings=591, final_activation=final_act_fun)
            elif model_name == 'LSTM':
                lr = 0.001
                net = LSTM(input_size, hidden_size, n_layers, len(y_columns),
                           num_embeddings=591, final_activation=final_act_fun)
            elif model_name == 'GRU':
                lr = 0.001
                net = GRU(input_size, hidden_size, n_layers, len(y_columns),
                          num_embeddings=591, final_activation=final_act_fun)
            else:
                raise ValueError("Invalid model name...")
            model_handler = ModelHandler(net, lr=lr, loss=loss_fun, sch_step=sch_step//valid_gap)

            arch = f"{model_name}_ipsize{input_size}_hsize{hidden_size}_numlayer{n_layers}_lr{lr}"
            weight_path = f"{args.model_dir}/{args.split}/{args.mode}/{arch}_seed{split_seed}_weight"
            log_path = f"{args.model_dir}/{args.split}/{args.mode}/{arch}_{split_seed}_log"
            print('weight_path', weight_path)
            print('log_path', log_path)

            checkpoint = torch.load(weight_path)
            net.load_state_dict(checkpoint['net_weight'])
            dict_list = []
            for _, (id_, x, y, w, row) in enumerate(tqdm(test_loader)):
                assert x.size(0) == 1, 'batch size should be 1'
                pred = model_handler.inference(x)

                row = {key: value[0].item() if not isinstance(value[0], str) else value[0] for key, value in row.items()}
                row['Pred'] = pred.item()
                dict_list.append(row)

            df = pd.DataFrame(dict_list)
            df.to_csv(f"{args.csv_dir}/{args.split}/{args.mode}/89_{arch}_{split_seed}.csv", index=False)


if __name__ == '__main__':
    csv_list = [f'../CSV/Data/mol_length_{i}.csv' for i in [8, 9]]
    model_list = ['GRU', 'RNN', 'LSTM']
    pytorch_models_inference(model_list)
