import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV files into DataFrames
df = pd.read_csv('../CSV/Predictions/auc_random_mol6710_classification.csv')
df2 = pd.read_csv('../CSV/Predictions/auc_random_mol6710_regression.csv')
df3 = pd.read_csv('../CSV/Predictions/auc_random_mol6710_soft.csv')

model_order = ['RF', 'SVM', 'AttentiveFP', 'DMPNN', 'GAT', 'GCN', 'MPNN',
               'PAGTN', 'RNN', 'LSTM', 'GRU', 'ChemCeption', 'ImageMol']  # Replace with actual model names

print(df['model'])
# Convert 'model' column to categorical with a specified order
df['model'] = pd.Categorical(df['model'], categories=model_order[::-1], ordered=True)
df2['model'] = pd.Categorical(df2['model'], categories=model_order[::-1], ordered=True)
df3['model'] = pd.Categorical(df3['model'], categories=model_order[::-1], ordered=True)
print(df['model'])
# # Sort the DataFrames accordingly
# df = df.sort_values('model')
# df2 = df2.sort_values('model')
# df3 = df3.sort_values('model')

# Create the plot
plt.figure(figsize=(9, 6))

# Plot data from the second CSV (regression)
plt.plot(df2['auc'], df2['model'], '^', markersize=8, markerfacecolor='none', markeredgecolor='k', label='Regression')
plt.errorbar(df2['auc'], df2['model'], xerr=df2['std'], fmt='none', color='k', capsize=5, capthick=1, elinewidth=1)

# Plot data from the first CSV (classification)
plt.plot(df['auc'], df['model'], 'o', markersize=8, markerfacecolor='none', markeredgecolor='b', label='Classification (Binary)')
plt.errorbar(df['auc'], df['model'], xerr=df['std'], fmt='none', color='b', capsize=5, capthick=1, elinewidth=1)

# Plot data from the second CSV (regression)
plt.plot(df3['auc'], df3['model'], 'x', markersize=8, markerfacecolor='none', markeredgecolor='r', label='Classification (Soft)')
plt.errorbar(df3['auc'], df3['model'], xerr=df3['std'], fmt='none', color='r', capsize=5, capthick=1, elinewidth=1)

plt.gca().set_yticks(df2['model'][::-1])
plt.gca().set_yticklabels(df2['model'][::-1])
plt.gca().invert_yaxis()

# Adding labels and title
plt.tick_params(axis='both', direction='in', labelsize=14)
# plt.ylabel('Model', fontsize=18)
plt.yticks(rotation=45)
plt.xlabel('ROC-AUC', fontsize=18)

# plt.title('AUC with Standard Deviation for Classification and Regression Models')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
plt.legend(fontsize=16)

# Show the plot
plt.tight_layout()
plt.savefig('Figure2.pdf', dpi=300, bbox_inches='tight')
plt.show()
