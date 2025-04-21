import os.path
import argparse
from typing import Any, Dict, List


def model_trainer(
    model: Any,
    save_dir: str,
    train_data: Any,
    valid_data: Any,
    metrics: List[Any],
    score_name: str,
    transformers: List[Any],
    text: str,
    args: argparse.Namespace
) -> Any:
    """
    Trains a model and performs early stopping based on validation performance.

    Parameters:
    model (Any): The model to be trained (e.g., a DeepChem model).
    save_dir (str): Directory where the best model checkpoint will be saved.
    train_data (Any): Dataset for training (must be compatible with model.fit()).
    valid_data (Any): Dataset for validation (used to monitor performance).
    metrics (List[Any]): List of evaluation metric objects.
    score_name (str): Name of the metric to monitor for early stopping.
    transformers (List[Any]): List of data transformers applied to datasets.
    text (str): Label or description for logging output.
    args (argparse.Namespace): Argument namespace. Must contain:
        - mode (str): One of ['regression', 'classification', 'soft'].
        - n_epoch (int): Number of training epochs.
        - patience (int): Number of epochs without improvement before stopping.

    Returns:
    Any: The best model (with weights restored from checkpoint if applicable).
    """
    if args.mode == 'regression':
        current_loss = float('inf')
    elif args.mode == 'classification' or args.mode == 'soft':
        current_loss = float('-inf')  # Use negative infinity for classification
    else:
        raise ValueError("Invalid mode. Mode should be 'regression' or 'classification'.")

    current_patient = 0
    # best_model = deepcopy(model)

    for epoch in range(args.n_epoch):
        loss = model.fit(train_data, nb_epoch=1, checkpoint_interval=0)
        # print(model.predict(train_data)[:5, :])
        # print(train_data.y[:5, :])
        valid_metrics = model.evaluate(valid_data, metrics, transformers)
        valid_loss = valid_metrics[score_name]
        print(f"{text} - Epoch {epoch}: Train loss: {loss}, Validation metric: {score_name} {valid_loss}")

        if (args.mode == 'regression' and valid_loss < current_loss) or \
           ((args.mode == 'classification' or args.mode == 'soft') and valid_loss > current_loss):
            current_loss = valid_loss
            current_patient = 0
            # best_model = deepcopy(model)  # Update best model
            model.save_checkpoint(model_dir=save_dir, max_checkpoints_to_keep=1)
            print(f"New best model saved with validation loss: {valid_loss}, save model: {save_dir}")
        else:
            current_patient += 1

        if current_patient > args.patience:
            print(f"Early stopping. Validation loss {current_loss} did not improve for {current_patient} epochs.")
            break

    if os.path.exists(f"{save_dir}/checkpoint1.pt"):    # for other models
        model.restore(f"{save_dir}/checkpoint1.pt")
    else:                                               # for ChemCeption
        model.restore()
    return model
