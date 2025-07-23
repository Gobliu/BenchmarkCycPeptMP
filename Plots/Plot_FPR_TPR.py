import pandas as pd
import matplotlib.pyplot as plt

reg_df = pd.read_csv('roc_curve_data_regression_GCN_seed1.csv')
sft_df = pd.read_csv('roc_curve_data_soft_GCN_seed1.csv')

plt.figure(figsize=(8, 6))
plt.plot(sft_df.Threshold[1:], sft_df.FPR[1:], color='r', label=f'Soft FPR', lw=2)
plt.plot(sft_df.Threshold[1:], sft_df.TPR[1:], color='tomato', label=f'Soft TPR', lw=2)
plt.plot(reg_df.Threshold[2:-1], reg_df.FPR[2:-1], color='b', label=f'Regression FPR', lw=2)
plt.plot(reg_df.Threshold[2:-1], reg_df.TPR[2:-1], color='dodgerblue', label=f'Regression TPR', lw=2)

# plt.plot(sft_df.Threshold[1:], sft_df.TPR[1:] - sft_df.FPR[1:], color='r', label=f'Soft TPR - FPR', lw=2)
# plt.plot(reg_df.Threshold[2:-1], reg_df.TPR[2:-1] - reg_df.FPR[2:-1], color='b', label=f'Regression TPR - FPR', lw=2)

plt.xlabel('Threshold', fontsize=20)
# plt.xticks(ticks=np.array([1, 5, 10, 15, 20]))  # every integer from 1 to max_k
# plt.ylim(0.6, 1.05)
plt.ylabel('TPR - FPR', fontsize=20)
# plt.title('Chemical Diversity via Top-k Similarity', fontsize=24)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(fontsize=18, loc='lower left', bbox_to_anchor=(0.01, 0.01))
plt.tick_params(axis='both', direction='in', labelsize=16)
plt.tight_layout()
# plt.savefig("FPR_TPR_Diff.pdf", dpi=300)
plt.show()
