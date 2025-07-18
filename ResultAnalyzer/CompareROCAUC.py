import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


def plot_roc_auc():
    # Create figure
    plt.figure(figsize=(10, 8))

    # Define colors for different modes (expand as needed)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(mode_list)))

    for mode, color in zip(mode_list, colors):
        # Initialize storage for ROC curves
        all_fpr = []
        all_tpr = []
        auc_scores = []
        mean_fpr = np.linspace(0, 1, 100)

        for seed in seed_list:
            csv = f'../CSV/Predictions/{split}/{mode}/{model}_seed{seed}.csv'
            df = pd.read_csv(csv)

            # Process labels (more concise version)
            true = (df.Normalized_PAMPA / 2 + 0.5).clip(0, 1).round().astype(int)
            if mode == 'regression':
                pred = df[f'Pred_{seed}'] / 2 + 0.5
            else:
                pred = df[f'Pred_{seed}']

            # Compute ROC
            fpr, tpr, thresholds = metrics.roc_curve(true, pred)

            # Combine into a DataFrame
            roc_df = pd.DataFrame({
                'FPR': fpr,
                'TPR': tpr,
                'Threshold': thresholds
            })

            # Save to CSV
            roc_df.to_csv(f'roc_curve_data_{mode}_{model}_seed{seed}.csv', index=False)

            roc_auc = metrics.auc(fpr, tpr)

            # Store results
            all_fpr.append(fpr)
            all_tpr.append(tpr)
            auc_scores.append(roc_auc)

        # Compute mean ROC
        interp_tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)]

        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[0], mean_tpr[-1] = 0.0, 1.0  # Ensure proper bounds

        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)
        std_tpr = np.std(interp_tprs, axis=0)

        # Plot with confidence interval
        plt.plot(mean_fpr, mean_tpr, color=color,
                 label=f'{mode} (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
                 lw=2)
        plt.fill_between(mean_fpr,
                         mean_tpr - std_tpr,
                         mean_tpr + std_tpr,
                         color=color, alpha=0.1)

    # Format plot
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.tick_params(axis='both', direction='in', labelsize=14)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title(f'Mean ROC curves by {model} model, {split} split', fontsize=14)
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(True, alpha=0.3)

    # Save and show
    plt.tight_layout()
    plt.savefig(f'ROC-AUC.pdf', dpi=300)
    plt.show()


if __name__ == '__main__':
    seed_list = list(range(1, 11))
    split = 'random'
    mode_list = ['regression', 'soft']
    model = 'GCN'
    plot_roc_auc()
