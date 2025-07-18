import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# df = pd.read_csv('DMPNN_Random_Split.csv')
# print(len(df))
# r2_list = []
# for seed in range(1, 10):
#     sub_df = df[df[f'split{seed}'] == 'train']
#     print(len(sub_df))
#     true = sub_df.Normalized_PAMPA
#     r2 = r2_score(true * 2 - 6, sub_df[f'Pred_{seed}'] * 2 - 6)
#     r2_list.append(r2)
#
# print('r2', np.mean(r2_list), np.std(r2_list, ddof=1), r2_list)
# quit()


df = pd.read_csv('DMPNN_Random_Split.csv')
seed = 1  # single seed used
sub_df = df[df[f'split{seed}'] == 'test']

# Create two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

# Plot 1: Original data
true1 = sub_df.Normalized_PAMPA * 2 - 6
pred1 = sub_df[f'Pred_{seed}'] * 2 - 6
r2_1 = r2_score(true1, pred1)

ellipse1 = Ellipse(xy=(-8, -6.35), width=0.4, height=1.4,
                   edgecolor='blue', linestyle='--', facecolor='none', linewidth=2)
axes[0].add_patch(ellipse1)
axes[0].scatter(true1, pred1, color='k', alpha=0.6, label=f'$R^2$ = {r2_1:.3f}')
min_val = min(true1.min(), pred1.min())
max_val = max(true1.max(), pred1.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--')
axes[0].set_xlabel('Experimental Permeability', fontsize=18)
axes[0].set_ylabel('Predicted Permeability', fontsize=18)
axes[0].set_title('Original Test Set', fontsize=20)
axes[0].grid(True, linestyle='--', alpha=0.3)
# axes[0].legend(loc='upper left', fontsize=14)
axes[0].text(0.05, 0.95, f'$R^2$ = {r2_1:.3f}', transform=axes[0].transAxes,
             fontsize=18, verticalalignment='top', horizontalalignment='left', fontstyle='italic')
axes[0].tick_params(axis='both', direction='in', labelsize=14)

# Plot 2: Filtered again
sub_df2 = sub_df[~((sub_df['Normalized_PAMPA'] == -1) & (sub_df[f'Pred_{seed}'] > -0.5))]
print(len(sub_df), len(sub_df2))
true2 = sub_df2.Normalized_PAMPA * 2 - 6
pred2 = sub_df2[f'Pred_{seed}'] * 2 - 6
r2_2 = r2_score(true2, pred2)

axes[1].scatter(true2, pred2, color='k', alpha=0.6, label=f'$R^2$ = {r2_2:.3f}')
# min_val = min(true2.min(), pred2.min())
# max_val = max(true2.max(), pred2.max())
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--')
axes[1].set_xlabel('Experimental Permeability', fontsize=18)
axes[1].set_title('Filtered Test Set', fontsize=20)
axes[1].grid(True, linestyle='--', alpha=0.3)
# axes[1].legend(loc='upper left', fontsize=14)
axes[1].text(0.05, 0.95, f'$R^2$ = {r2_2:.3f}', transform=axes[1].transAxes,
             fontsize=18, verticalalignment='top', horizontalalignment='left', fontstyle='italic')
axes[1].tick_params(axis='both', direction='in', labelsize=14)

labels = ['A', 'B']
for i, ax in enumerate(axes):
    ax.text(-0.05, 1.1, labels[i], transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.savefig(f'ScatterPlot.pdf', dpi=300)
plt.show()
