import os
import shutil
import sys
from sklearn.metrics import mean_squared_error

# Dynamically append project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from CustomizedDateLoader import *
from DeepChemModels.ModelFeatureGenerator import generate_model_feature
from DeepChemModels.ModelTrainer import model_trainer
from Utils import set_seed, get_combined_args, get_csv_list

from deepchem.models.losses import SparseSoftmaxCrossEntropy

metric_dict = {'classification': [dc.metrics.Metric(dc.metrics.f1_score), 'f1_score'],
               'regression': [dc.metrics.Metric(dc.metrics.score_function.rms_score), 'rms_score'],
               'soft': [dc.metrics.Metric(dc.metrics.f1_score), 'f1_score']}

task_dict = {'classification': ['Binary'],
             'regression': ['Normalized_PAMPA'],
             'soft': ['Soft_Label']}


def main():
    args = get_combined_args()
    assert args.model in ['DMPNN', 'GCN', 'GAT', 'AttentiveFP', 'MPNN', 'PAGTN', 'ChemCeption'],\
        f'DeepChemMain.py does not handle model {args.model}...'
    csv_list = get_csv_list(args)

    print(f'==== training {args.model} model ====')

    m_score, score_name = metric_dict[args.mode]
    # print(rms, rms_score)
    # quit()
    tasks = task_dict[args.mode]
    for split_seed in range(1, args.repeat+1):
        set_seed(123 * split_seed ** 2)
        feat, model = generate_model_feature(args.model, len(tasks), args)
        print(model.loss, model.optimizer, model.optimizer.learning_rate)
        if hasattr(model.optimizer, 'scheduler'):
            print("The model is using a learning rate schedule.")
        else:
            print("The model is using a fixed learning rate.")
        weight_dir = f"{args.model_dir}/{args.split}/{args.mode}/{args.model}"
        os.makedirs(weight_dir, exist_ok=True)
        csv_dir = f"{args.csv_dir}/{args.split}/{args.mode}"
        os.makedirs(csv_dir, exist_ok=True)
        print('==== pre train ====')
        if args.mode == 'regression':
            _, datasets_pre_train, transformers_pre_train = dc.molnet.load_delaney(
                featurizer=feat, splitter='random'
            )
        elif args.mode == 'classification' or args.mode == 'soft':
            _, datasets_pre_train, transformers_pre_train = dc.molnet.load_bbbp(
                featurizer=feat, splitter='random'
            )

        train_pre_train, valid_pre_train, test_pre_train = datasets_pre_train

        if len(tasks) > 1:
            train_pre_train = convert_multitask(train_pre_train, len(tasks))
            valid_pre_train = convert_multitask(valid_pre_train, len(tasks))
            # test_pre_train = convert_multitask(test_pre_train, len(tasks))

        model = model_trainer(
            model, weight_dir,
            train_pre_train, valid_pre_train,
            metrics=[m_score], score_name=score_name, transformers=transformers_pre_train,
            text=f'Pre-Training with seed {split_seed}', args=args
        )

        print('==== train cyclic peptide data ====')
        loader = dc.data.CSVLoader(
            tasks=tasks, feature_field="SMILES",
            id_field="Original_Name_in_Source_Literature", featurizer=feat
        )

        if args.split == 'scaffold':
            assert len(csv_list) == 3, 'Expect a list of [train_list, valid_list, test_list]'
            data = data_loader_separate(csv_list, loader)
        elif args.split == 'random':
            data = data_loader_all_in_one(csv_list, split_seed, loader)

        train_cp, valid_cp, test_cp, test_df = data['train'], data['valid'], data['test'], data['test_df']

        if args.mode == 'classification' or args.mode == 'soft':
            train_cp, valid_cp, test_cp = adjust_class_weights(train_cp, valid_cp, test_cp, [1.])
        elif len(tasks) > 1 and args.mode == 'regression':
            train_weights = [1.] + [0.1 for _ in tasks[1:]]
            train_cp = adjust_task_weight(train_cp, train_weights)
            valid_weights = [1.] + [0.0 for _ in tasks[1:]]
            valid_cp = adjust_task_weight(valid_cp, valid_weights)
            # print(train_cp.w)
            # print(valid_cp.w)

        if args.mode == 'soft':
            train_cp = p2distribution(train_cp)
            valid_cp = soft_label2hard(valid_cp)
            test_cp = soft_label2hard(test_cp)

        model = model_trainer(
            model, f"{args.model_dir}/{args.split}/{args.mode}/{args.model}", train_cp, valid_cp,
            metrics=[m_score], score_name=score_name, transformers=[],
            text=f'Training permeability with seed {split_seed}', args=args
        )

        if os.path.exists(f"{weight_dir}/checkpoint1.pt"):      # for other models
            print(f"{weight_dir}/checkpoint1.pt -> {weight_dir}/checkpoint_seed{split_seed}.pt")
            os.rename(f"{weight_dir}/checkpoint1.pt", f"{weight_dir}/checkpoint_seed{split_seed}.pt")
        else:
            print(f"{weight_dir} -> {weight_dir}_seed{split_seed}")
            if os.path.exists(f"{weight_dir}_seed{split_seed}"):
                shutil.rmtree(f"{weight_dir}_seed{split_seed}")
            os.rename(f"{weight_dir}", f"{weight_dir}_seed{split_seed}")
        print('Confirm valid loss', model.evaluate(valid_cp, [m_score])[score_name])

        # print(model.predict(test_cp).shape, model.predict(test_cp)[0, ...])

        if args.mode == 'regression':
            # print(test_cp.y)
            # print(model.predict(test_cp))
            if len(tasks) > 1:
                for i, t in enumerate(tasks):
                    test_df[f'Pred_{t}'] = model.predict(test_cp)[:, i].squeeze()
                    print(f'rmse {t} in train',
                          mean_squared_error(train_cp.y[:, i], model.predict(train_cp).squeeze()[:, i],
                                             squared=False))
                    print(f'rmse {t} in test',
                          mean_squared_error(test_cp.y[:, i], model.predict(test_cp).squeeze()[:, i].squeeze(),
                                             squared=False))
            else:
                test_df[f'Pred_{split_seed}'] = model.predict(test_cp).squeeze()
                print(f'rmse in train', mean_squared_error(train_cp.y, model.predict(train_cp).squeeze(),
                                                           squared=False))
                print(f'rmse in test', mean_squared_error(test_cp.y, model.predict(test_cp).squeeze(),
                                                          squared=False))

        elif args.mode == 'classification' or args.mode == 'soft':
            test_df[f'Pred_{split_seed}'] = model.predict(test_cp).squeeze()[:, 1]
        # test_df[f'True_{split_seed}'] = test_cp.y

        test_csv_path = f"{csv_dir}/{args.model}_seed{split_seed}.csv"
        print('Saving csv of test data to', test_csv_path)
        test_df.to_csv(test_csv_path, index=False)
        # quit()

        # ====== for ChemCeption, check inference on peptide length 8/9 =======
        # if args.model == 'ChemCeption':
        #     infer_df = pd.read_csv('../CSV/Data/mol_length_8and9.csv')
        #     infer_cp = loader.create_dataset('../CSV/Data/mol_length_8and9.csv')
        #     print('Infer loss', model.evaluate(infer_cp, [m_score])[score_name])
        #     if args['mode'] == 'regression':
        #         infer_df[f'Pred_{split_seed}'] = model.predict(infer_cp).squeeze()
        #     elif args['mode'] == 'classification':
        #         infer_df[f'Pred_{split_seed}'] = model.predict(infer_cp).squeeze()[:, 1]
        #     infer_csv_path = f"{csv_dir}/89_{args.model}_seed{split_seed}.csv"
        #     print('Saving csv of test data to', infer_csv_path)
        #     infer_df.to_csv(infer_csv_path, index=False)


if __name__ == '__main__':
    main()
