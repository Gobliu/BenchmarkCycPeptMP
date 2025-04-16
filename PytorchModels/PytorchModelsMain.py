import sys
import torch.nn
import yaml
from tqdm import tqdm
import pandas as pd

sys.path.append('../')

from Models import RNN, LSTM, GRU
from DataSet import DataSetSMILES
from ModelHandler import ModelHandler
from Utils import set_seed


def prepare_pretrain_data_loaders(args, dict_path, batch_size):
    torch.cuda.empty_cache()
    kwargs = {'num_workers': 1, 'pin_memory': False} if torch.cuda.is_available() else {}

    if args['mode'] == 'regression':
        train_data = DataSetSMILES(['../CSV/PreTrainData/Delaney_Train_Set.csv'], dict_path,
                                   x_column='SMILES', y_columns=['normalized_solubility'], id_column='Compound ID')
        valid_data = DataSetSMILES(['../CSV/PreTrainData/Delaney_Validation_Set.csv'], dict_path,
                                   x_column='SMILES', y_columns=['normalized_solubility'], id_column='Compound ID')
    else:
        train_data = DataSetSMILES(['../CSV/PreTrainData/BBBP_Train_Set.csv'], dict_path, max_token=250,
                                   x_column='SMILES', y_columns=['Binary'], id_column='num')
        valid_data = DataSetSMILES(['../CSV/PreTrainData/BBBP_Validation_Set.csv'], dict_path, max_token=250,
                                   x_column='SMILES', y_columns=['Binary'], id_column='num')
        pw, nw = train_data.get_weight()
        train_data.set_weight(pw, nw)
        valid_data.set_weight(pw, nw)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size, shuffle=False, **kwargs)

    return train_loader, valid_loader


def prepare_data_loaders(csv_list, split_seed, args, dict_path, x_column, y_columns, batch_size):
    torch.cuda.empty_cache()
    kwargs = {'num_workers': 1, 'pin_memory': False} if torch.cuda.is_available() else {}

    if args['split'] == 'scaffold':
        split_csv_list = csv_list
    elif args['split'] == 'random':
        ip_df = pd.concat([pd.read_csv(csv) for csv in csv_list], ignore_index=True)
        grouped = ip_df.groupby(f'split{split_seed}')
        for group_name, group_df in grouped:
            group_df.to_csv(f'temp_{group_name}.csv', index=False)
        split_csv_list = [['temp_train.csv'], ['temp_valid.csv'], ['temp_test.csv']]

    train_data = DataSetSMILES(split_csv_list[0], dict_path, x_column=x_column, y_columns=y_columns)
    valid_data = DataSetSMILES(split_csv_list[1], dict_path, x_column=x_column, y_columns=y_columns)
    test_data  = DataSetSMILES(split_csv_list[2], dict_path, x_column=x_column, y_columns=y_columns)

    if args['mode'] in ['classification', 'soft']:
        pw, nw = train_data.get_weight()
        train_data.set_weight(pw, nw)
        valid_data.set_weight(pw, nw)
        test_data.set_weight(pw, nw)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size, shuffle=False, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_data, 1, shuffle=False, **kwargs)

    return train_loader, valid_loader, test_loader


def build_model(model_name, input_size, hidden_size, n_layers, y_dim, final_act_fun):
    if model_name == 'RNN':
        return RNN(input_size, hidden_size, n_layers, y_dim, num_embeddings=591, final_activation=final_act_fun), 0.0001
    elif model_name == 'LSTM':
        return LSTM(input_size, hidden_size, n_layers, y_dim, num_embeddings=591, final_activation=final_act_fun), 0.001
    elif model_name == 'GRU':
        return GRU(input_size, hidden_size, n_layers, y_dim, num_embeddings=591, final_activation=final_act_fun), 0.001
    else:
        raise ValueError("Invalid model name...")


def train_model(args, model_handler, net, train_loader, valid_loader, weight_path, log_path, valid_gap):
    current_loss = float('inf')
    patience_count = 0

    for i in range(args['n_epoch']):
        train_loss = model_handler.train(train_loader).item()
        if i % valid_gap == 0:
            valid_loss = model_handler.eval(valid_loader).item()
            if valid_loss < current_loss:
                current_loss = valid_loss
                best_epoch = i
                save_dic = {
                    "epoch": i,
                    "best_v_loss": current_loss,
                    "net_weight": net.state_dict(),
                    "optimizer_state_dict": model_handler.opt.state_dict(),
                }
                torch.save(save_dic, weight_path)
                patience_count = 0
            else:
                patience_count += valid_gap
            with open(log_path, "a") as log:
                log.write(f"epoch {i}, train {train_loss:.8f}, valid {valid_loss:.8f}, patience count {patience_count}\n")
                if patience_count > args["patience"]:
                    log.write(f"val_loss {current_loss} did not decrease for {args['patience']}.\n")
                    break

    with open(log_path, "a") as log:
        log.write(f"Best model in epoch {best_epoch} Val_loss: {current_loss}\n")

    return current_loss


def evaluate_and_save(net, model_handler, test_loader, weight_path, csv_out_path):
    checkpoint = torch.load(weight_path)
    net.load_state_dict(checkpoint['net_weight'])
    dict_list = []

    for _, (id_, x, y, w, row) in enumerate(tqdm(test_loader)):
        assert x.size(0) == 1
        pred = model_handler.inference(x)
        row = {key: value[0].item() if not isinstance(value[0], str) else value[0] for key, value in row.items()}
        row['Pred'] = pred.item()
        dict_list.append(row)

    pd.DataFrame(dict_list).to_csv(csv_out_path, index=False)


def main(csv_list):
    for split_seed in range(1, 11):
        set_seed(123 * split_seed ** 2)

        net, lr = build_model(model_name, input_size, hidden_size, n_layers, len(y_columns), final_act_fun)
        arch = f"{model_name}_ipsize{input_size}_hsize{hidden_size}_numlayer{n_layers}_lr{lr}"
        weight_path = f"{args['model_dir']}/{args['split']}/{args['mode']}/{arch}_seed{split_seed}_weight"
        log_path = f"{args['model_dir']}/{args['split']}/{args['mode']}/{arch}_{split_seed}_log"

        model_handler = ModelHandler(net, lr=lr, loss=loss_fun, sch_step=sch_step//valid_gap)

        print("======== start pre-training ========")
        train_loader_pre, valid_loader_pre = prepare_pretrain_data_loaders(args, dict_path, batch_size)

        train_model(args, model_handler, net, train_loader_pre, valid_loader_pre, weight_path, log_path, valid_gap)
        print("======== start training ========")
        train_loader, valid_loader, test_loader = prepare_data_loaders(
            csv_list, split_seed, args, dict_path, x_column, y_columns, batch_size
        )

        train_model(args, model_handler, net, train_loader, valid_loader, weight_path, log_path, valid_gap)

        csv_out_path = f"{args['csv_dir']}/{args['split']}/{args['mode']}/{arch}_{split_seed}.csv"
        evaluate_and_save(net, model_handler, test_loader, weight_path, csv_out_path)


if __name__ == '__main__':
    yaml_config_path = '../Config.yaml'
    with open(yaml_config_path, 'r') as f:
        args = yaml.load(f, Loader=yaml.Loader)

    dict_path = 'vocab-large.txt'
    x_column = 'SMILES'
    batch_size = 16
    input_size = 128
    hidden_size = 128
    n_layers = 2
    valid_gap = 2
    sch_step = 100

    for model_name in ['LSTM', 'GRU', 'RNN']:
        if args['mode'] == 'regression':
            y_columns = ['Normalized_PAMPA']
            final_act_fun = torch.nn.Identity()
            loss_fun = torch.nn.L1Loss(reduction='none')
        elif args['mode'] == 'classification':
            y_columns = ['Binary']
            final_act_fun = torch.nn.Sigmoid()
            loss_fun = torch.nn.BCELoss(reduction='none')
        elif args['mode'] == 'soft':
            y_columns = ['Soft_Label']
            final_act_fun = torch.nn.Sigmoid()
            loss_fun = torch.nn.BCELoss(reduction='none')
        else:
            quit(f"Unknown mode:{args['mode']}")

        if args['split'] == 'scaffold':
            mol_length_list = [6, 7, 10]
            csv_list_ = [[f'../CSV/Data/mol_length_{i}_train.csv' for i in mol_length_list],
                         [f'../CSV/Data/mol_length_{i}_valid.csv' for i in mol_length_list],
                         [f'../CSV/Data/mol_length_{i}_test.csv' for i in mol_length_list]]
        elif args['split'] == 'random':
            csv_list_ = ["../CSV/Data/Random_Split.csv"]

        main(csv_list_)
