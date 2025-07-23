import time
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import rdFMCS, AllChem


def compute_sim_topk(df, top_k_range):
    start_time = time.time()
    mols = [Chem.MolFromSmiles(smiles) for smiles in df.SMILES]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]
    print('time (min):', (time.time() - start_time) / 60)
    sim_list = []
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i] + fps[i + 1:])
        sims.sort(reverse=True)
        sim_list.append(sims)
    sim_array = np.stack(sim_list, axis=0)  # Shape: (len(fps), len(fps)-1)
    # print(sim_array.shape)
    mean_list = []
    std_list = []
    for k in range(1, top_k_range+1):
        sim_top_k = sim_array[:, :k]
        top_k_mean = np.mean(sim_top_k, axis=1)
        # print(sim_top_k.shape, top_k_mean.shape)
        mean_list.append(np.mean(top_k_mean))
        std_list.append(np.std(top_k_mean))
        # print('time (min):', (time.time() - start_time) / 60)
    return np.array(mean_list), np.array(std_list)


if __name__ == '__main__':
    scaffold_train = ['../CSV/Data/mol_length_6_train.csv',
                      '../CSV/Data/mol_length_7_train.csv',
                      '../CSV/Data/mol_length_10_train.csv']
    scaffold_test = ['../CSV/Data/mol_length_6_test.csv',
                     '../CSV/Data/mol_length_7_test.csv',
                     '../CSV/Data/mol_length_10_test.csv']

    scaffold_train_df = pd.concat([pd.read_csv(i) for i in scaffold_train], ignore_index=True)
    scaffold_test_df = pd.concat([pd.read_csv(i) for i in scaffold_test], ignore_index=True)

    df = pd.read_csv('../CSV/Data/Random_Split.csv')
    random_train_df = df[df['split1'] == 'train'].copy()
    random_test_df = df[df['split1'] == 'test'].copy()

    max_k = 50
    plt.figure(figsize=(12, 9))
    x = list(range(1, max_k + 1))

    sim_mean, sim_std = compute_sim_topk(random_train_df, max_k)

    # plt.errorbar(x, sim_mean, yerr=sim_std, fmt='o-', color='b', capsize=10, label='Random Split Train')
    #
    # sim_mean, sim_std = compute_mcs(random_test_df, max_k)
    # plt.errorbar(x, sim_mean, yerr=sim_std, fmt='o--', color='green', capsize=10, label='Random Split Test')
    #
    # sim_mean, sim_std = compute_mcs(scaffold_train_df, max_k)
    # plt.errorbar(x, sim_mean, yerr=sim_std, fmt='^-', color='r', capsize=10, label='Scaffold Split Train')
    #
    # sim_mean, sim_std = compute_mcs(scaffold_test_df, max_k)
    # plt.errorbar(x, sim_mean, yerr=sim_std, fmt='^--', color='purple', capsize=10, label='Scaffold Split Test')

    sim_mean, sim_std = compute_sim_topk(random_train_df, max_k)
    plt.plot(x, sim_mean, color='b', lw=2, label='Train (Random Split)')
    plt.fill_between(x, sim_mean - sim_std, sim_mean + sim_std, color='b', alpha=0.1)

    # Random Split Test
    sim_mean, sim_std = compute_sim_topk(random_test_df, max_k)
    plt.plot(x, sim_mean, color='b', lw=2, linestyle='--', label='Test (Random Split)')
    plt.fill_between(x, sim_mean - sim_std, sim_mean + sim_std, color='green', alpha=0.1)

    # Scaffold Split Train
    sim_mean, sim_std = compute_sim_topk(scaffold_train_df, max_k)
    plt.plot(x, sim_mean, color='r', lw=2, label='Train (Scaffold Split)')
    plt.fill_between(x, sim_mean - sim_std, sim_mean + sim_std, color='r', alpha=0.1)

    # Scaffold Split Test
    sim_mean, sim_std = compute_sim_topk(scaffold_test_df, max_k)
    plt.plot(x, sim_mean, color='r', lw=2, linestyle='--', label='Test (Scaffold Split)')
    plt.fill_between(x, sim_mean - sim_std, sim_mean + sim_std, color='purple', alpha=0.1)

    plt.xlabel('Top-k Most Similar Neighbors', fontsize=20)
    plt.xticks(ticks=np.array([1, 5, 10, 15, 20]))  # every integer from 1 to max_k
    plt.ylim(0.6, 1.05)
    plt.ylabel('Average Top-k Tanimoto Similarity', fontsize=20)
    plt.title('Chemical Diversity via Top-k Similarity', fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=20, loc='lower left', bbox_to_anchor=(0.01, 0.01))
    plt.tick_params(axis='both', direction='in', labelsize=16)
    plt.tight_layout()
    # plt.savefig("TopK_Similarity_Plot.pdf", dpi=300)
    plt.show()
