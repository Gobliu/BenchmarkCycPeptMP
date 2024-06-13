from copy import deepcopy


def trainer_regression(model, n_epoch, patience, train_data, valid_data, metrics, transformers, text):
    current_loss = float('inf')
    current_patient = 0
    best_model = deepcopy(model)
    for i in range(n_epoch):
        loss = model.fit(train_data, nb_epoch=1, checkpoint_interval=0)
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


def trainer_classification(model, n_epoch, patience, train_data, valid_data, metrics, transformers, text):
    current_loss = 0        # check!!!, spent hours
    current_patient = 0
    best_model = deepcopy(model)
    for i in range(n_epoch):
        loss = model.fit(train_data, nb_epoch=1, checkpoint_interval=0)
        valid_loss = model.evaluate(valid_data, metrics, transformers)['prc_auc_score']
        print(text, 'epoch', i, 'loss', loss, 'valid loss', valid_loss)
        if valid_loss > current_loss:
            current_loss = valid_loss
            current_patient = 0
            best_model = deepcopy(model)
        else:
            current_patient += 1

        if current_patient > patience:
            print(f"val_loss {current_loss} did not decrease for {current_patient} epochs consequently.")
            break
    return best_model
