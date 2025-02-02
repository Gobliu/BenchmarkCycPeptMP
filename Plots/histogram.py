# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
#
# # # Load your dataset
# # df = pd.read_csv('../CSV/Data/Random_Split.csv', low_memory=False)
# # df['PAMPA_clipped'] = df['PAMPA'].clip(lower=-8, upper=-4)
# #
# # df_89 = pd.read_csv('../CSV/Data/mol_length_89.csv', low_memory=False)
# # df_89['PAMPA_clipped'] = df_89['PAMPA'].clip(lower=-8, upper=-4)
# #
# # # Plot the histogram with Seaborn
# # plt.figure(figsize=(8, 6))
# # sns.histplot(df['PAMPA_clipped'].dropna(), binwidth=0.5, color='blue', stat='percent', label='PAMPA', kde=False)
# # sns.histplot(df_89['PAMPA_clipped'].dropna(), binwidth=0.5, color='red', stat='percent', label='89', kde=False)
# # plt.title('Histogram of PAMPA Values (Clipped to [-8, -4])')
# # plt.xlabel('PAMPA')
# # plt.ylabel('Percentage')
# # plt.show()
#
# # import pandas as pd
# # import seaborn as sns
# # import matplotlib.pyplot as plt
#
# # Load the datasets
# df_random_split = pd.read_csv('../CSV/Data/Random_Split.csv', low_memory=False)
# df_mol_length_89 = pd.read_csv('../CSV/Data/mol_length_89.csv', low_memory=False)
#
# # Clip the 'PAMPA' values to the range [-8, -4]
# df_random_split['PAMPA_clipped'] = df_random_split['PAMPA'].clip(lower=-8, upper=-4)
# df_mol_length_89['PAMPA_clipped'] = df_mol_length_89['PAMPA'].clip(lower=-8, upper=-4)
#
# # Add a 'Dataset' column to distinguish the datasets
# df_random_split['Dataset'] = 'Random_Split'
# df_mol_length_89['Dataset'] = 'Mol_Length_89'
#
# # Combine the datasets
# df_combined = pd.concat([df_random_split[['PAMPA_clipped', 'Dataset']],
#                          df_mol_length_89[['PAMPA_clipped', 'Dataset']]])
# cutoff = -6
# # Plot the histograms side by side
# # Plot the histograms side by side with percentages
# plt.figure(figsize=(10, 6))
# ax = plt.gca()
# plt.xlim(df_combined['PAMPA_clipped'].min() - 0.1, df_combined['PAMPA_clipped'].max() + 0.1)
# ax.axvspan(df_combined['PAMPA_clipped'].min() - 0.1, cutoff, facecolor='lightblue', alpha=0.3)
# ax.axvspan(cutoff, df_combined['PAMPA_clipped'].max() + 0.1, facecolor='lightgreen', alpha=0.3)
# # ax = sns.histplot(data=df_combined, x='PAMPA_clipped', hue='Dataset', multiple='dodge', common_norm=False,
# #                   binwidth=0.5, shrink=0.8, stat='percent', edgecolor='black')
# sns.histplot(data=df_combined, x='PAMPA_clipped', hue='Dataset', multiple='dodge',
#              binwidth=0.5, shrink=0.8, stat='percent', common_norm=False, edgecolor='black', zorder=1, ax=ax)
# ax.tick_params(axis='both', direction='in', labelsize=14)  # Adjust 'labelsize' as needed
# ax.axvline(cutoff, color='red', linestyle='--', linewidth=1, zorder=2)
#
#
# plt.title('Side-by-Side Histograms of PAMPA Values (Clipped to [-8, -4])')
# plt.xlabel('PAMPA')
# plt.ylabel('Percentage')
# plt.legend(title='Dataset', loc='upper left', fontsize=12)
# plt.show()
#
#
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
df_random_split = pd.read_csv('../CSV/Data/Random_Split.csv', low_memory=False)
df_mol_length_89 = pd.read_csv('../CSV/Data/mol_length_89.csv', low_memory=False)

# Clip the 'PAMPA' values to the range [-8, -4]
df_random_split['PAMPA_clipped'] = df_random_split['PAMPA'].clip(lower=-8, upper=-4)
df_mol_length_89['PAMPA_clipped'] = df_mol_length_89['PAMPA'].clip(lower=-8, upper=-4)

# Add a 'Dataset' column to distinguish the datasets
df_random_split['Dataset'] = 'Peptide Length 6/7/10'
df_mol_length_89['Dataset'] = 'Peptide Length 8/9'

# Combine the datasets
df_combined = pd.concat([df_random_split[['PAMPA_clipped', 'Dataset']],
                         df_mol_length_89[['PAMPA_clipped', 'Dataset']]])

# Check the combined DataFrame
print(df_combined.head())
print(df_combined['Dataset'].unique())

# Define cutoff and plot
cutoff = -6
plt.figure(figsize=(10, 6))
ax = plt.gca()
plt.xlim(df_combined['PAMPA_clipped'].min() - 0.1, df_combined['PAMPA_clipped'].max() + 0.1)
ax.axvspan(df_combined['PAMPA_clipped'].min() - 0.1, cutoff, facecolor='lightblue', alpha=0.3)
ax.axvspan(cutoff, df_combined['PAMPA_clipped'].max() + 0.1, facecolor='lightgreen', alpha=0.3)

# Plot histograms
sns.histplot(data=df_combined, x='PAMPA_clipped', hue='Dataset', multiple='dodge',
             binwidth=0.5, shrink=0.8, stat='percent', common_norm=False, edgecolor='black', zorder=1, ax=ax)

# Add cutoff line
ax.axvline(cutoff, color='red', linestyle='--', linewidth=1, zorder=2)
ax.tick_params(axis='both', direction='in', labelsize=14)  # Adjust 'labelsize' as needed
# Add legend explicitly
handles, labels = ax.get_legend_handles_labels()
if handles:  # Check if there are any handles to display
    legend = ax.legend(title=None, loc='upper left', fontsize=16)  # Set title=None
    for text in legend.get_texts():
        text.set_fontsize(16)  # Explicitly set fontsize for legend labels
else:
    print("Warning: No handles found for legend.")

# Add titles and labels
# plt.title('Side-by-Side Histograms of PAMPA Values (Clipped to [-8, -4])')
plt.xlabel('Permeability', fontsize=16)
plt.ylabel('Peptide Frequency (%)', fontsize=16)
plt.savefig('histogram_pampa.png', dpi=300, bbox_inches='tight')
plt.show()
