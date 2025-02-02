import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV files into DataFrames
df = pd.read_csv('../CSV/Predictions/auc_random_mol6710_classification.csv')
df2 = pd.read_csv('../CSV/Predictions/auc_random_mol6710_regression.csv')
df3 = pd.read_csv('../CSV/Predictions/auc_random_mol6710_soft.csv')

# Create the plot
plt.figure(figsize=(10, 6))

# Plot data from the first CSV (classification)
plt.plot(df['auc'], df['model'], 'o', markersize=8, markerfacecolor='none', markeredgecolor='b', label='Classification')
plt.errorbar(df['auc'], df['model'], xerr=df['std'], fmt='none', color='b', capsize=5, capthick=1, elinewidth=1)

# Plot data from the second CSV (regression)
plt.plot(df2['auc'], df2['model'], '^', markersize=8, markerfacecolor='none', markeredgecolor='k', label='Regression')
plt.errorbar(df2['auc'], df2['model'], xerr=df2['std'], fmt='none', color='k', capsize=5, capthick=1, elinewidth=1)

# Plot data from the second CSV (regression)
plt.plot(df3['auc'], df3['model'], 'x', markersize=8, markerfacecolor='none', markeredgecolor='r', label='Soft')
plt.errorbar(df3['auc'], df3['model'], xerr=df3['std'], fmt='none', color='r', capsize=5, capthick=1, elinewidth=1)

# Adding labels and title
plt.ylabel('Model')
plt.xlabel('AUC')
plt.title('AUC with Standard Deviation for Classification and Regression Models')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
