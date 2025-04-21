import os
import glob
import shutil
from argparse import Namespace
from DispatcherMain import dispatch

ML_MODELS = ['RF', 'SVM']
PYTORCH_MODELS = ['GRU', 'RNN', 'LSTM']
DEEPCHEM_MODELS = ['DMPNN', 'AttentiveFP', 'PAGTN', 'GAT', 'GCN', 'MPNN', 'ChemCeption']

ALL_MODELS = ML_MODELS + PYTORCH_MODELS + DEEPCHEM_MODELS
ALL_SPLITS = ['random', 'scaffold']
ALL_MODES = ['regression', 'classification', 'soft']

model_dir = "./SavedModel"
csv_dir = "./Predictions"


def test_dispatcher():
    for model in ALL_MODELS:
        for split in ALL_SPLITS:
            for mode in ALL_MODES:
                os.makedirs(f"{model_dir}/{split}/{mode}", exist_ok=True)
                os.makedirs(f"{csv_dir}/{split}/{mode}", exist_ok=True)
                args = Namespace(
                    model=model,
                    split=split,
                    mode=mode,
                    n_epoch=1,
                    batch_size=64,
                    patience=2,
                    model_dir="./SavedModel",
                    csv_dir="./Predictions",
                    repeat=1,
                )
                print(f"Testing: model={model}, split={split}, mode={mode}")
                dispatch(args)

    shutil.rmtree(model_dir, ignore_errors=True)
    shutil.rmtree(csv_dir, ignore_errors=True)
    print("Temporary directories cleaned up.")

    for file in glob.glob('temp_*.csv'):
        if os.path.isfile(file):
            os.remove(file)


if __name__ == '__main__':
    test_dispatcher()
