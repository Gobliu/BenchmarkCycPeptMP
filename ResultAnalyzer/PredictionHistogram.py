import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_side_by_side_histogram():
    plt.figure(figsize=(10, 6))
    all_preds_global = []

    for label in [0, 1]:
        # First pass: gather all predictions to set common bin range
        mode_preds = {}
        for mode in mode_list:
            all_preds = []
            for seed in seed_list:
                csv = f'../CSV/Predictions/{split}/{mode}/{model}_seed{seed}.csv'
                df = pd.read_csv(csv)
                df = df[df['Binary'] == label]
                if mode == 'regression':
                    preds = df[f'Pred_{seed}'] / 2 + 0.5
                else:
                    preds = df[f'Pred_{seed}']
                all_preds.extend(preds)
            mode_preds[mode] = np.array(all_preds)
            all_preds_global.extend(all_preds)

        # Define shared bin edges
        bin_width = 0.02
        min_pred = min(all_preds_global)
        max_pred = max(all_preds_global)
        bins = np.arange(min_pred, max_pred + bin_width, bin_width)
        bin_centers = bins[:-1] + bin_width / 2

        # Plot each mode side-by-side using bar chart
        n_modes = len(mode_list)
        bar_width = bin_width / n_modes
        colors = plt.cm.rainbow(np.linspace(0, 1, n_modes))

        # for i, mode in enumerate(mode_list):
        #     hist, _ = np.histogram(mode_preds[mode], bins=bins, density=True)
        #     offsets = bin_centers - (bin_width / 2) + i * bar_width
        #     plt.bar(offsets, hist, width=bar_width, alpha=0.8, label=mode, color=colors[i], edgecolor='black')
        #
        # plt.xlabel('Predicted Value', fontsize=12)
        # plt.ylabel('Density', fontsize=12)
        # plt.title(f'Prediction Distribution label {label})', fontsize=14)
        # plt.legend(fontsize=10)
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        # plt.savefig(f'PredictHistogram_{label}.png', dpi=300)
        # plt.show()

        for i, mode in enumerate(mode_list):
            # Compute cumulative histogram (CDF)
            hist, _ = np.histogram(mode_preds[mode], bins=bins, density=True)
            cdf = np.cumsum(hist)
            cdf = cdf / cdf[-1]  # normalize to [0, 1]

            plt.plot(bin_centers, cdf, label=mode, color=colors[i], linewidth=2)

        plt.xlabel('Predicted Value', fontsize=12)
        plt.ylabel('Cumulative Probability', fontsize=12)
        plt.title(f'Cumulative Prediction Distribution (Label {label})', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'CumulativePredictHistogram_{label}.png', dpi=300)
        plt.show()


if __name__ == '__main__':
    seed_list = list(range(1, 11))
    split = 'random'
    mode_list = ['regression', 'soft', 'classification']
    model = 'GCN'
    plot_side_by_side_histogram()
