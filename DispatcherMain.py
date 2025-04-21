import os
import subprocess
from Utils import get_combined_args

# Define supported models by category
ML_MODELS = ['RF', 'SVM']
PYTORCH_MODELS = ['GRU', 'RNN', 'LSTM']
DEEPCHEM_MODELS = ['DMPNN', 'AttentiveFP', 'PAGTN', 'GAT', 'GCN', 'MPNN', 'ChemCeption']

# Mapping from model name to corresponding script
MODEL_TO_SCRIPT = {model: './ClassicalML/MLMain.py' for model in ML_MODELS}
MODEL_TO_SCRIPT.update({model: './PytorchModels/PytorchModelsMain.py' for model in PYTORCH_MODELS})
MODEL_TO_SCRIPT.update({model: './DeepChemModels/DeepChemModelsMain.py' for model in DEEPCHEM_MODELS})


def dispatch(args):
    model_name = args.model
    script_name = MODEL_TO_SCRIPT.get(model_name)

    if args.mode == 'soft' and model_name in ML_MODELS + ['ChemCeption']:
        print(f'!!! This model {model_name} has no classification with soft label')
        return

    if script_name is None:
        raise ValueError(f"Unsupported model: {model_name}")

    command = ['python', script_name] + [f'--{k}={v}' for k, v in vars(args).items()]
    print(f"Dispatching to {script_name} with model '{model_name}'")
    print("Running command:", ' '.join(command))
    subprocess.run(command)


def main():
    args = get_combined_args()
    dispatch(args)


if __name__ == '__main__':
    main()
