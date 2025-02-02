import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


df = pd.read_csv('../CSV/Data/CycPeptMPDB_Peptide_All.csv', low_memory=False)
sns.set(font_scale=1.4)
# Define bins and labels
bins = [1, 5, 6, 7, 9, 10, 15]
labels = ['2-5', '6', '7', '8-9', '10', '11-15']
# explode = (0, 0.05, 0.05, 0, 0.05, 0)

# Categorize 'monomer_length' into bins
df['length_group'] = pd.cut(df['Monomer_Length'], bins=bins, labels=labels, right=True)

# Calculate the frequency of each group
group_counts = df['length_group'].value_counts().sort_index()

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(group_counts, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 14},
        startangle=90, counterclock=False,
        colors=sns.color_palette('Set2'))
plt.title('Distribution of Peptide Lengths')
plt.ylabel('')  # Hide the y-label
plt.savefig('pie_length.png', dpi=300, bbox_inches='tight')
# plt.show()

labels = ['PAMPA',  'MDCK', 'Caco2', 'RRCK']
group_counts = [6655, 79, 689, 496]
plt.figure(figsize=(8, 8))
plt.pie(group_counts, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 14},
        startangle=90, counterclock=False,
        colors=sns.color_palette('Set3'))
plt.title('Distribution of Permeability Assays')
plt.ylabel('')  # Hide the y-label
plt.savefig('pie_assay.png', dpi=300, bbox_inches='tight')
plt.show()
