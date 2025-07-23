import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_top_down_histograms():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    all_preds_global = []

    for label_idx, label in enumerate([0, 1]):
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
    bin_width = 0.1
    bins = np.arange(0, 1 + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width / 2

    # Plot each label on separate axes
    for label_idx, label in enumerate([0, 1]):
        ax = axes[label_idx]

        if label == 0:
            msg = 'Negative Samples'
        else:
            msg = 'Positive Samples'

        # Recompute mode_preds just for this label
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

        # Plot bars with gap
        gap_ratio = 0.2
        n_modes = len(mode_list)
        total_gap = gap_ratio * bin_width
        bar_width = (bin_width - total_gap) / n_modes
        colors = plt.cm.rainbow(np.linspace(0, 1, n_modes))

        print('bins', bins)
        for i, mode in enumerate(mode_list):
            hist, _ = np.histogram(mode_preds[mode], bins=bins, density=False)
            print(msg, mode, hist, sum(hist))
            offsets = bin_centers - (bin_width / 2) + i * bar_width + (total_gap / 2)
            ax.bar(offsets, hist, width=bar_width, alpha=0.8, label=mode,
                   color=colors[i], edgecolor='black')

        ax.set_ylabel('Count', fontsize=20)
        ax.set_title(f'Prediction Distribution of {msg}', fontsize=20)
        ax.legend(fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=20)

    labels = ['A', 'B']
    for i, ax in enumerate(axes):
        ax.text(-0.1, 1.1, labels[i], transform=ax.transAxes,
                fontsize=20, fontweight='bold', va='top', ha='right')

    axes[0].set_xlabel('Predicted Value', fontsize=20)
    axes[1].set_xlabel('Predicted Value', fontsize=20)
    plt.tight_layout()
    plt.savefig('PredictHistogram_TopDown.pdf', dpi=300)
    plt.show()


if __name__ == '__main__':
    seed_list = list(range(1, 11))
    split = 'random'
    mode_list = ['regression', 'soft']
    model = 'GCN'
    plot_top_down_histograms()
