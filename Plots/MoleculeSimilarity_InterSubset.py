import time
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def compute_max_sim(src_df, tgt_df):
    start_time = time.time()
    src_mols = [Chem.MolFromSmiles(smiles) for smiles in src_df.SMILES]
    src_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in src_mols]

    tgt_mols = [Chem.MolFromSmiles(smiles) for smiles in tgt_df.SMILES]
    tgt_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in tgt_mols]

    print('Similarity computation time (min):', (time.time() - start_time) / 60)

    sim_list = []
    for fp in src_fps:
        sims = max(DataStructs.BulkTanimotoSimilarity(fp, tgt_fps))
        sim_list.append(sims)
    return sim_list


if __name__ == '__main__':
    # Load scaffold-based split
    scaffold_train = ['../CSV/Data/mol_length_6_train.csv',
                      '../CSV/Data/mol_length_7_train.csv',
                      '../CSV/Data/mol_length_10_train.csv']
    scaffold_valid = ['../CSV/Data/mol_length_6_valid.csv',
                      '../CSV/Data/mol_length_7_valid.csv',
                      '../CSV/Data/mol_length_10_valid.csv']
    scaffold_test = ['../CSV/Data/mol_length_6_test.csv',
                     '../CSV/Data/mol_length_7_test.csv',
                     '../CSV/Data/mol_length_10_test.csv']

    scaffold_train_df = pd.concat([pd.read_csv(i) for i in scaffold_train], ignore_index=True)
    scaffold_valid_df = pd.concat([pd.read_csv(i) for i in scaffold_valid], ignore_index=True)
    scaffold_test_df = pd.concat([pd.read_csv(i) for i in scaffold_test], ignore_index=True)

    # Load random split
    df = pd.read_csv('../CSV/Data/Random_Split.csv')
    random_train_df = df[df['split1'] == 'train'].copy()
    random_valid_df = df[df['split1'] == 'valid'].copy()
    random_test_df = df[df['split1'] == 'test'].copy()

    # Histogram parameters
    bin_width = 0.05
    bins = np.arange(0.2, 1 + bin_width + 1e-6, bin_width)
    bin_centers = bins[:-1] + bin_width / 2

    gap_ratio = 0.2
    n_groups = 2  # random + scaffold
    total_gap = gap_ratio * bin_width
    bar_width = (bin_width - total_gap) / n_groups
    colors = plt.cm.rainbow(np.linspace(0, 1, n_groups))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    labels = ['Random Split', 'Scaffold Split']

    sim_lists_valid = [compute_max_sim(random_valid_df, random_train_df),
                       compute_max_sim(scaffold_valid_df, scaffold_train_df)]

    for i, (label, max_sims) in enumerate(zip(labels, sim_lists_valid)):
        hist, _ = np.histogram(max_sims, bins=bins)
        offsets = bin_centers - (bin_width / 2) + i * bar_width + (total_gap / 2)
        axes[0].bar(offsets, hist, width=bar_width, alpha=0.8, label=label, color=colors[i], edgecolor='black')

    axes[0].set_ylabel('Count', fontsize=20)
    axes[0].set_xlabel('Tanimoto Similarity (Morgan Fingerprints)', fontsize=20)
    axes[0].set_title('Validation Sample to Training Data', fontsize=20)
    axes[0].legend(fontsize=20)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='both', labelsize=20)

    sim_lists_test = [compute_max_sim(random_test_df, random_train_df),
                      compute_max_sim(scaffold_test_df, scaffold_train_df)]

    # print(min(sim_lists_test[0]), min(sim_lists_valid[0]), sum(sim_lists_test[0]), sum(sim_lists_test[1]))
    # print(min(sim_lists_test[1]), min(sim_lists_valid[1]))

    for i, (label, max_sims) in enumerate(zip(labels, sim_lists_test)):
        hist, _ = np.histogram(max_sims, bins=bins)
        print(sum(hist), np.sum(np.array(max_sims) < 0.2))
        offsets = bin_centers - (bin_width / 2) + i * bar_width + (total_gap / 2)
        axes[1].bar(offsets, hist, width=bar_width, alpha=0.8, label=label, color=colors[i], edgecolor='black')

    # axes[1].set_ylabel('Count', fontsize=20)
    axes[1].set_xlabel('Tanimoto Similarity (Morgan Fingerprints)', fontsize=20)
    axes[1].set_title('Test Sample to Training Data', fontsize=20)
    axes[1].legend(fontsize=20)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='both', labelsize=20)

    labels = ['A', 'B']
    for i, ax in enumerate(axes):
        ax.text(-0.1, 1.1, labels[i], transform=ax.transAxes,
                fontsize=20, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.savefig('Similarity_hist.pdf', dpi=300)
    plt.show()
