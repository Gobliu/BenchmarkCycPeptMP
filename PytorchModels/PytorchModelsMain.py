import os
import sys
sys.path.append('../')

import torch

from Models import *
from DataSet import DataSetSMILES
from ModelHandler import ModelHandler
from DeepChemModels.CustomizedDateLoader import data_loader_all_in_one
from Utils import set_seed


def main():
    torch.cuda.empty_cache()
    kwargs = {'num_workers': 1, 'pin_memory': False} if torch.cuda.is_available() else {}
    train_data = DataSetSMILES(train_csv_list, dict_path, x_column=x_column, y_column=y_column)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=False)
    valid_data = DataSetSMILES(valid_csv_list, dict_path, x_column=x_column, y_column=y_column)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)

    model = RNN(128, 512, 3, len(y_column), num_embeddings=591, final_activation=torch.nn.Identity())
    model_handler = ModelHandler(model, lr=lr, loss=torch.nn.L1Loss())

    # arch = f"{args['model_name']}_dmodel{args['dmodel']}_mlplayer{args['mlp_layer']}_encoderlayer{args['encoder_layer']}"
    # weight_path = f"{args['op_dir']}/{arch}_weight"
    # print('~~~~~~~', weight_path, os.path.exists(weight_path))
    # log_path = f"{args['op_dir']}/{arch}_log"
    #
    # if args["load_weight"] and os.path.exists(weight_path):
    #     checkpoint = torch.load(weight_path)
    #     model.g_net_wrapper.g_net.load_state_dict(checkpoint["g_net_weight"])
    #     mydevice.load(model.g_net_wrapper.g_net)
    #     model.opt.load_state_dict(checkpoint["optimizer_state_dict"])
    #     # trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     start_epoch = checkpoint["epoch"] + 1
    #     best_v_loss = checkpoint["best_v_loss"]
    #     print('~~~~~~~~~~~~~~~~~~~~~loading weight', weight_path, 'epoch', start_epoch, 'loss', best_v_loss)
    # else:
    #     start_epoch = 0
    #     best_v_loss = sys.maxsize
    #
    # patience_count = 0
    # best_epoch = 0

    print("======== start training ========")
    for i in range(start_epoch, args['end_epoch']):
        # print('training...')
        train_loss = model.train(loader.train_loader)
        # train_loss = model.train(loader.val_loader)
        if i % args['valid_gap'] == 0:
            # print('validating...')
            # model.g_net_wrapper.g_net.test()
            valid_loss = model.eval(loader.val_loader, print_worst=None)
            # valid_loss = model.eval(loader.train_loader, print_worst=None)
            if valid_loss < best_v_loss:
                best_v_loss = valid_loss
                best_epoch = i
                save_dic = {
                    "epoch": i,
                    "best_v_loss": best_v_loss,
                    "g_net_weight": model.g_net_wrapper.g_net.state_dict(),
                    "optimizer_state_dict": model.opt.state_dict(),
                }
                torch.save(save_dic, weight_path)
                patience_count = 0
            else:
                patience_count += 1
            message = f"epoch {i}, train {train_loss:.8f}, valid {valid_loss:.8f}"
            print(message)
            with open(log_path, "a") as log:
                log.write(message + "\n")
                if patience_count > args["patience"]:
                    print(f"val_loss {best_v_loss} did not decrease for {args['patience']}.")
                    break

    with open(log_path, "a") as log:
        log.write(f"Best model in epoch {best_epoch} Val_loss: {best_v_loss}\n")
    print("training ended.")

    for i in range(100):
        train_loss = model_handler.train(train_loader)
        valid_loss = model_handler.eval(valid_loader)
        print(train_loss.item(), valid_loss.item())


if __name__ == '__main__':
    dict_path = 'vocab.txt'
    x_column = ['SMILES']
    y_column = ['Normalized_PAMPA']
    batch_size = 16
    lr = 0.0001
    mol_length_list = [6, 7, 10]
    train_csv_list = [f'../CSV/Data/mol_length_{i}_train.csv' for i in mol_length_list]
    valid_csv_list = [f'../CSV/Data/mol_length_{i}_valid.csv' for i in mol_length_list]
    main()
