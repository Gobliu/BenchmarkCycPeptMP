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


def main(csv_list):
    torch.cuda.empty_cache()
    kwargs = {'num_workers': 1, 'pin_memory': False} if torch.cuda.is_available() else {}

    for split_seed in range(1, 11):
        set_seed(123 * split_seed ** 2)
        if args['split'] == 'scaffold':
            assert len(csv_list) == 3, 'Expect a list of [train_list, valid_list, test_list]'
            split_csv_list = csv_list
        elif args['split'] == 'random':
            df_list = []
            for csv in csv_list:
                print(csv)
                df_list.append(pd.read_csv(csv))
            ip_df = pd.concat(df_list, ignore_index=True)
            grouped = ip_df.groupby(f'split{split_seed}')
            for group_name, group_df in grouped:
                group_df.to_csv(f'temp_{group_name}.csv', index=False)
            split_csv_list = [['temp_train.csv'], ['temp_valid.csv'], ['temp_test.csv']]

        print('csv list', csv_list)
        train_data = DataSetSMILES(split_csv_list[0], dict_path, x_column=x_column, y_columns=y_columns)
        valid_data = DataSetSMILES(split_csv_list[1], dict_path, x_column=x_column, y_columns=y_columns)
        test_data = DataSetSMILES(split_csv_list[2], dict_path, x_column=x_column, y_columns=y_columns)
        if args['mode'] == 'classification' or args['mode'] == 'soft':
            pw, nw = train_data.get_weight()
            train_data.set_weight(pw, nw)
            valid_data.set_weight(pw, nw)
            test_data.set_weight(pw, nw)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size, shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_data, 1, shuffle=False, **kwargs)
        # print(train_data[0])
        # quit()

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
        weight_path = f"{args['model_dir']}/{args['split']}/{args['mode']}/{arch}_seed{split_seed}_weight"
        log_path = f"{args['model_dir']}/{args['split']}/{args['mode']}/{arch}_{split_seed}_log"
        print('weight_path', weight_path)
        print('log_path', log_path)

        current_loss = float('inf')

        print("======== start training ========")
        patience_count = 0
        for i in range(args['n_epoch']):
            # print('training...')
            train_loss = model_handler.train(train_loader).item()
            if i % valid_gap == 0:
                # print('validating...')
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
                message = f"epoch {i}, train {train_loss:.8f}, valid {valid_loss:.8f}, patience count {patience_count}"
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
        # loss = torch.nn.L1Loss()
        dict_list = []
        for _, (id_, x, y, w, row) in enumerate(tqdm(test_loader)):
            assert x.size(0) == 1, 'batch size should be 1'
            pred = model_handler.inference(x)
            # print('id', id_, 'y', y, 'pred', pred, 'loss', loss(pred, y))
            # quit()
            row = {key: value[0].item() if not isinstance(value[0], str) else value[0] for key, value in row.items()}
            row['Pred'] = pred.item()
            dict_list.append(row)

            # # series = pd.concat([series, pd.Series({'Pred': pred.item()})])
            # try:
            #     df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            # except UnboundLocalError:
            #     df = s.to_frame(name='Column_Name')

        df = pd.DataFrame(dict_list)
        df.to_csv(f"{args['csv_dir']}/{args['split']}/{args['mode']}/{arch}_{split_seed}.csv", index=False)
        # df.to_csv(f"{arch}_{split_seed}.csv", index=False)
        # quit()


if __name__ == '__main__':
    yaml_config_path = '../Config.yaml'
    with open(yaml_config_path, 'r') as f:
        args = yaml.load(f, Loader=yaml.Loader)
    dict_path = 'vocab-large.txt'
    x_column = 'SMILES'         # DO NOT convert it into a list
    batch_size = 16
    input_size = 128
    hidden_size = 128
    n_layers = 2
    valid_gap = 2
    sch_step = 100
    for model_name in ['GRU', 'RNN', 'LSTM']:
    # model_name = 'GRU'
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
