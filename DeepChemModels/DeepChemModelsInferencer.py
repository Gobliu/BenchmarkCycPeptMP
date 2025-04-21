import os
import sys

# Dynamically append project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from DeepChemModels.CustomizedDateLoader import *
from DeepChemModels.ModelFeatureGenerator import generate_model_feature
from DeepChemModels.DeepChemModelsMain import task_dict
from Utils import get_combined_args


def deepchem_models_inferencer(m_names):
    args = get_combined_args()
    for model_name in m_names:
        print(f'==== training {model_name} model ====')

        # rms, rms_score = rms_dict[args['mode']]
        tasks = task_dict[args.mode]

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
            test_cp = p2distribution(test_cp)
            for split_seed in range(1, args.repeat+1):
                if model_name != 'ChemCeption':
                    wp = f"{args.model_dir}/{args.split}/{args.mode}/{model_name}/checkpoint_seed{split_seed}.pt"
                    print('Loading', wp)
                    model.restore(wp)
                else:
                    model = dc.models.ChemCeption(
                        n_tasks=len(tasks), mode=args.mode,
                        model_dir=f"{args.model_dir}/{args.split}/{args.mode}/{model_name}_seed{split_seed}",
                        batch_size=args.batch_size)
                    print(model.model_dir)
                    model.restore()
                if args.mode == 'regression':
                    test_df[f'Pred_{split_seed}'] = model.predict(test_cp).squeeze()
                elif args.mode == 'classification' or args.mode == 'soft':
                    print(model.predict(test_cp).shape)
                    test_df[f'Pred_{split_seed}'] = model.predict(test_cp).squeeze()[:, 1]
            test_csv_path = f"{args.csv_dir}/{args.split}/{args.mode}/{model_name}_{csv_name}"
            # print(test_csv_path)
            print('Saving csv of test data to', test_csv_path)
            test_df.to_csv(test_csv_path, index=False)


if __name__ == '__main__':

    rms_dict = {'classification': [dc.metrics.Metric(dc.metrics.prc_auc_score), 'prc_auc_score'],
                'regression': [dc.metrics.Metric(dc.metrics.score_function.rms_score), 'rms_score'],
                'soft': [dc.metrics.Metric(dc.metrics.prc_auc_score), 'prc_auc_score']}

    csv_list = [f'../CSV/Data/mol_length_{i}.csv' for i in [8, 9]]

    print('Working on csv list:', csv_list)
    model_list = ['DMPNN', 'GCN', 'GAT', 'MPNN', 'PAGTN', 'AttentiveFP', 'ChemCeption']
    deepchem_models_inferencer(model_list[:])
