from scipy import stats
import pandas as pd


def compute_p_value_and_stats(df, split, mode, model_1, model_2, metric):
    """
    Computes the p-value, mean, and standard deviation for a given metric between two models using the specified split and mode.

    Parameters:
    df (DataFrame): The DataFrame containing model performance data.
    split (str): The type of data split (e.g., 'random', 'scaffold').
    mode (str): The task mode (e.g., 'regression').
    model_1 (str): The name of the first model.
    model_2 (str): The name of the second model.
    metric (str): The target metric to compare (e.g., 'mae', 'rmse').

    Returns:
    dict: A dictionary with p-value, mean, and standard deviation for both models.
    """
    # Filter the DataFrame based on the split and mode
    filtered_df = df[(df['split'] == split) & (df['mode'] == mode)]

    # Extract the metric values for the two models
    model_1_data = filtered_df[filtered_df['model'] == model_1][metric]
    model_2_data = filtered_df[filtered_df['model'] == model_2][metric]

    # Compute mean and standard deviation for both models
    model_1_mean = model_1_data.mean()
    model_1_std = model_1_data.std()
    model_2_mean = model_2_data.mean()
    model_2_std = model_2_data.std()

    # Perform the t-test
    t_stat, p_value = stats.ttest_ind(model_1_data, model_2_data)

    return {
        'model_1': {
            'model': model_1,
            'mean': model_1_mean,
            'std': model_1_std,
        },
        'model_2': {
            'model': model_2,
            'mean': model_2_mean,
            'std': model_2_std,
        },
        'p_value': p_value
    }


if __name__ == '__main__':
    split = 'random'
    mode = 'classification'
    metric_csv = f'./CSV/Predictions/metric_{mode}.csv'
    model_1 = 'AttentiveFP'
    model_2 = 'DMPNN'
    metric = 'auc'
    df = pd.read_csv(metric_csv)
    results = compute_p_value_and_stats(df, split, mode, model_1, model_2, metric)
    print(results)
