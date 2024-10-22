import sys

import yaml
from tqdm import tqdm
import pandas as pd

sys.path.append('../')

from Models import *
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
            splited_csv_list = csv_list
        elif args['split'] == 'random':
            df_list = []
            for csv in csv_list:
                print(csv)
                df_list.append(pd.read_csv(csv))
            ip_df = pd.concat(df_list, ignore_index=True)
            grouped = ip_df.groupby(f'split{split_seed}')
            for group_name, group_df in grouped:
                group_df.to_csv(f'temp_{group_name}.csv', index=False)
            splited_csv_list = [['temp_train.csv'], ['temp_valid.csv'], ['temp_test.csv']]

        print('csv list', csv_list)
        train_data = DataSetSMILES(splited_csv_list[0], dict_path, x_column=x_column, y_column=y_column)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, **kwargs)
        valid_data = DataSetSMILES(splited_csv_list[1], dict_path, x_column=x_column, y_column=y_column)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size, shuffle=False, **kwargs)
        test_data = DataSetSMILES(splited_csv_list[2], dict_path, x_column=x_column, y_column=y_column)
        test_loader = torch.utils.data.DataLoader(test_data, 1, shuffle=False, **kwargs)

        if model_name == 'RNN':
            net = RNN(input_size, hidden_size, n_layers, len(y_column),
                      num_embeddings=591, final_activation=torch.nn.Identity())
        else:
            raise ValueError("Invalid model name...")
        model_handler = ModelHandler(net, lr=lr, loss=torch.nn.L1Loss(), sch_step=sch_step//valid_gap)

        arch = f"{model_name}_ipsize{input_size}_hsize{hidden_size}_numlayer{n_layers}_lr{lr}"
        weight_path = f"../{args['model_dir']}/{args['split']}/{args['mode']}/{arch}_seed{split_seed}_weight"
        log_path = f"../{args['model_dir']}/{args['split']}/{args['mode']}/{arch}_{split_seed}_log"
        print('weight_path', weight_path)
        print('log_path', log_path)

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
            if i % valid_gap == 0:
                # print('validating...')
                valid_loss = model_handler.eval(valid_loader).item()
                if (args['mode'] == 'regression' and valid_loss < current_loss) or \
                        ((args['mode'] == 'classification' or args['mode'] == 'soft') and valid_loss > current_loss):
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
        # loss = torch.nn.L1Loss()
        dict_list = []
        for _, (id_, x, y, row) in enumerate(tqdm(test_loader)):
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
        df.to_csv(f"../{args['csv_dir']}/{args['split']}/{args['mode']}/{arch}_{split_seed}.csv", index=False)


if __name__ == '__main__':
    yaml_config_path = '../Config.yaml'
    with open(yaml_config_path, 'r') as f:
        args = yaml.load(f, Loader=yaml.Loader)
    dict_path = 'vocab-large.txt'
    x_column = ['SMILES']
    y_column = ['Normalized_PAMPA']
    batch_size = 16
    lr = 0.0001
    input_size = 64
    hidden_size = 64
    n_layers = 2
    valid_gap = 5
    model_name = 'RNN'
    sch_step = 100

    if args['split'] == 'scaffold':
        mol_length_list = [6, 7, 10]
        csv_list_ = [[f'../CSV/Data/mol_length_{i}_train.csv' for i in mol_length_list],
                    [f'../CSV/Data/mol_length_{i}_valid.csv' for i in mol_length_list],
                    [f'../CSV/Data/mol_length_{i}_test.csv' for i in mol_length_list]]
    elif args['split'] == 'random':
        csv_list_ = ["../CSV/Data/Random_Split.csv"]
    main(csv_list_)
