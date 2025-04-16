import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV files into DataFrames
df = pd.read_csv('MAE.csv')

model_order = ['RF', 'SVM', 'AttentiveFP', 'DMPNN', 'GAT', 'GCN', 'MPNN',
               'PAGTN', 'RNN', 'LSTM', 'GRU', 'ChemCeption', 'ImageMol']  # Replace with actual model names

print(df['model'])
# Convert 'model' column to categorical with a specified order
df['model'] = pd.Categorical(df['model'], categories=model_order[::-1], ordered=True)
print(df['model'])
# # Sort the DataFrames accordingly
# df = df.sort_values('model')
# df2 = df2.sort_values('model')
# df3 = df3.sort_values('model')

# Create the plot
plt.figure(figsize=(9, 6))

# Plot data from the second CSV (regression)
plt.plot(df['mae'], df['model'], 'o', markersize=8, markerfacecolor='none', markeredgecolor='k', label='Regression')
plt.errorbar(df['mae'], df['model'], xerr=df['std'], fmt='none', color='k', capsize=5, capthick=1, elinewidth=1)

plt.axvline(x=0.098, color='#1f77b4', linestyle='--', linewidth=2, label='Intra-record (PAMPA)')     # blue
plt.axvline(x=0.206, color='#ff7f0e', linestyle='--', linewidth=2, label='Intra-report (PAMPA)')     # orange
plt.axvline(x=0.867, color='#2ca02c', linestyle='--', linewidth=2, label='Inter-report (PAMPA)')     # green

plt.gca().set_yticks(df['model'][::-1])
plt.gca().set_yticklabels(df['model'][::-1])
plt.gca().invert_yaxis()

# Adding labels and title
plt.tick_params(axis='both', direction='in', labelsize=14)
# plt.ylabel('Model', fontsize=18)
plt.yticks(rotation=45)
plt.xlabel('Mean Absolute Error (MAE)', fontsize=16)

# plt.title('AUC with Standard Deviation for Classification and Regression Models')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
plt.legend(fontsize=14, loc='upper left', bbox_to_anchor=(0.55, 1))

# Show the plot
plt.tight_layout()
plt.savefig('Figure2.pdf', dpi=300, bbox_inches='tight')
plt.show()
