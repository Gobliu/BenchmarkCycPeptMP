import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

df = pd.read_csv('../CSV/Predictions/random/regression/mp_logp_tpsa/DMPNN_seed1.csv')

# Create two subplots side by side
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Plot 1: Original data
true_pampa = df.Normalized_PAMPA
pred_pampa = df.Pred_Normalized_PAMPA
r2_pampa = r2_score(true_pampa, pred_pampa)
mae_pampa = mean_absolute_error(true_pampa, pred_pampa)

axes[0].scatter(true_pampa, pred_pampa, color='k', alpha=0.6, label=f'$R^2$={r2_pampa:.3f} MAE={mae_pampa:.3f}')
# custom_label = Line2D([0], [0], linestyle='none', color='black', label=f'$R^2$={r2_pampa:.3f} MAE={mae_pampa:.3f}')
axes[0].text(0.05, 0.95, f'$R^2$ = {r2_pampa:.3f}  MAE = {mae_pampa:.3f}', transform=axes[0].transAxes,
             fontsize=20, verticalalignment='top', horizontalalignment='left', fontstyle='italic')
min_val = min(true_pampa.min(), pred_pampa.min())
max_val = max(true_pampa.max(), pred_pampa.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--')
axes[0].set_xlabel('Normalized Experimental Permeability', fontsize=20)
axes[0].set_ylabel('Predicted Permeability', fontsize=20)
axes[0].set_title('Permeability', fontsize=24)
axes[0].grid(True, linestyle='--', alpha=0.3)
# axes[0].legend(loc='upper left', fontsize=14)
# axes[0].legend(handles=[custom_label], loc='upper left', fontsize=20)
axes[0].tick_params(axis='both', direction='in', labelsize=16)

true_logp = df.Normalized_MolLogP
pred_logp = df.Pred_Normalized_MolLogP
r2_logp = r2_score(true_logp, pred_logp)
mae_logp = mean_absolute_error(true_logp, pred_logp)

axes[1].scatter(true_logp, pred_logp, color='k', alpha=0.6, label=f'$R^2$ = {r2_logp:.3f} MAE={mae_logp:.3f}')
# custom_label = Line2D([0], [0], linestyle='none', color='black', label=f'$R^2$={r2_logp:.3f} MAE={mae_logp:.3f}')
axes[1].text(0.05, 0.95, f'$R^2$ = {r2_logp:.3f}  MAE = {mae_logp:.3f}', transform=axes[1].transAxes,
             fontsize=20, verticalalignment='top', horizontalalignment='left', fontstyle='italic')
min_val = min(true_logp.min(), pred_logp.min())
max_val = max(true_logp.max(), pred_logp.max())
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--')
axes[1].set_xlabel('Normalized LogP', fontsize=20)
axes[1].set_ylabel('Predicted LogP', fontsize=20)
axes[1].set_title('LogP', fontsize=24)
axes[1].grid(True, linestyle='--', alpha=0.3)
# axes[1].legend(loc='upper left', fontsize=14)
# axes[1].legend(handles=[custom_label], loc='upper left', fontsize=20)
axes[1].tick_params(axis='both', direction='in', labelsize=16)

true_tpsa = df.Normalized_TPSA
pred_tpsa = df.Pred_Normalized_TPSA
r2_tpsa = r2_score(true_tpsa, pred_tpsa)
mae_tpsa = mean_absolute_error(true_tpsa, pred_tpsa)

axes[2].scatter(true_tpsa, pred_tpsa, color='k', alpha=0.6, label=f'$R^2$ = {r2_tpsa:.3f} MAE={mae_tpsa:.3f}')
# custom_label = Line2D([0], [0], linestyle='none', color='black', label=f'$R^2$={r2_tpsa:.3f} MAE={mae_tpsa:.3f}')
axes[2].text(0.05, 0.95, f'$R^2$ = {r2_tpsa:.3f}  MAE = {mae_tpsa:.3f}', transform=axes[2].transAxes,
             fontsize=20, verticalalignment='top', horizontalalignment='left', fontstyle='italic')
min_val = min(true_tpsa.min(), pred_tpsa.min())
max_val = max(true_tpsa.max(), pred_tpsa.max())
axes[2].plot([min_val, max_val], [min_val, max_val], 'r--')
axes[2].set_xlabel('Normalized TPSA', fontsize=20)
axes[2].set_ylabel('Predicted TPSA', fontsize=20)
axes[2].set_title('TPSA', fontsize=24)
axes[2].grid(True, linestyle='--', alpha=0.3)
# axes[2].legend(loc='upper left', fontsize=14)
# axes[2].legend(handles=[custom_label], loc='upper left', fontsize=20)
axes[2].tick_params(axis='both', direction='in', labelsize=16)

labels = ['A', 'B', 'C']
for i, ax in enumerate(axes):
    ax.text(-0.1, 1.1, labels[i], transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.savefig(f'ScatterPlot3.pdf', dpi=300)
plt.show()
