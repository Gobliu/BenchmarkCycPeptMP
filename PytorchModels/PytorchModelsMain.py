import os
import sys

import pandas as pd

sys.path.append('../')

import torch
import yaml
import tqdm

from Models import *
from DataSet import DataSetSMILES
from ModelHandler import ModelHandler
from DeepChemModels.CustomizedDateLoader import data_loader_all_in_one
from Utils import set_seed


def main():
    torch.cuda.empty_cache()
    kwargs = {'num_workers': 1, 'pin_memory': False} if torch.cuda.is_available() else {}
    train_data = DataSetSMILES(train_csv_list, dict_path, x_column=x_column, y_column=y_column)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, **kwargs)
    valid_data = DataSetSMILES(valid_csv_list, dict_path, x_column=x_column, y_column=y_column)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size, shuffle=False, **kwargs)
    test_data = DataSetSMILES(test_csv_list, dict_path, x_column=x_column, y_column=y_column)
    test_loader = torch.utils.data.DataLoader(test_data, 1, shuffle=False, **kwargs)

    if model_name == 'RNN':
        net = RNN(input_size, hidden_size, n_layers, len(y_column),
                  num_embeddings=591, final_activation=torch.nn.Identity())
    else:
        raise ValueError("Invalid model name...")
    model_handler = ModelHandler(net, lr=lr, loss=torch.nn.L1Loss())

    arch = f"{model_name}_ipsize{input_size}_hsize{hidden_size}_numlayer{n_layers}_lr{lr}"
    weight_path = f"{args['model_dir']}/{args['split']}/{args['mode']}/{arch}_weight"
    log_path = f"{args['model_dir']}/{args['split']}/{args['mode']}/{arch}_log"

    if args['mode'] == 'regression':
        current_loss = float('inf')
    elif args['mode'] == 'classification' or args['mode'] == 'soft':
        current_loss = float('-inf')  # Use negative infinity for classification
    else:
        raise ValueError("Invalid mode. Mode should be 'regression' or 'classification'.")

    print("======== start training ========")
    patience_count = 0
    for i in range(args['n_epoch']):
        # print('training...')
        train_loss = model_handler.train(train_loader).item()
        if i % args['valid_gap'] == 0:
            # print('validating...')
            # model.g_net_wrapper.g_net.test()
            valid_loss = model_handler.eval(valid_loader).item()
            if (args['mode'] == 'regression' and valid_loss < current_loss) or \
                    ((args['mode'] == 'classification' or args['mode'] == 'soft') and valid_loss > current_loss):
                current_loss = valid_loss
                best_epoch = i
                save_dic = {
                    "epoch": i,
                    "best_v_loss": current_loss,
                    "net_weight": net.state_dict(),
                    "optimizer_state_dict": net.opt.state_dict(),
                }
                torch.save(save_dic, weight_path)
                patience_count = 0
            else:
                patience_count += args['valid_gap']
            message = f"epoch {i}, train {train_loss:.8f}, valid {valid_loss:.8f}"
            print(message)
            with open(log_path, "a") as log:
                log.write(message + "\n")
                if patience_count > args["patience"]:
                    print(f"val_loss {current_loss} did not decrease for {args['patience']}.")
                    break

    with open(log_path, "a") as log:
        log.write(f"Best model in epoch {best_epoch} Val_loss: {current_loss}\n")
    print("training ended.")

    checkpoint = torch.load(weight_path)
    net.load_state_dict(checkpoint['net_weight'])
    for _, (_, x, y, series) in enumerate(tqdm(test_loader)):
        assert x.size(0) == 1, 'batch size should be 1'
        pred = net.inference(x)
        series = pd.concat([series, pd.Series({'Pred': pred.item()})])
        df = df.append(series, ignore_index=True)
        # Create a DataFrame using the first Series
        df = pd.DataFrame([first_series])

if __name__ == '__main__':
    yaml_config_path = 'Config.yaml'
    with open(yaml_config_path, 'r') as f:
        args = yaml.load(f, Loader=yaml.Loader)
    dict_path = 'vocab.txt'
    x_column = ['SMILES']
    y_column = ['Normalized_PAMPA']
    batch_size = 16
    lr = 0.0001
    input_size = 128
    hidden_size = 512
    n_layers = 3
    model_name = 'RNN'
    mol_length_list = [6, 7, 10]
    train_csv_list = [f'../CSV/Data/mol_length_{i}_train.csv' for i in mol_length_list]
    valid_csv_list = [f'../CSV/Data/mol_length_{i}_valid.csv' for i in mol_length_list]
    test_csv_list = [f'../CSV/Data/mol_length_{i}_valid.csv' for i in mol_length_list]
    main()
