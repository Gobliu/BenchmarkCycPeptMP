import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import numpy as np


df = pd.read_csv('../CSV/Data/CycPeptMPDB_Peptide_All.csv')
# print(df.Monomer_Length)
# for i in df.Monomer_Length:
#     print(i, float(i))

color = seaborn.color_palette('bright')
count_data = df.Monomer_Length.value_counts()  # Replace 'cut' with your integer column
print(count_data)

# Plot a pie chart using matplotlib
# plt.figure(figsize=(8, 8))
# plt.pie(count_data, labels=count_data.index, autopct='%1.1f%%', startangle=90,
#         pctdistance=0.85, labeldistance=1.1)
# plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.title('Pie Chart of Cut Distribution')  # Replace with your title
# plt.show()

light_blue = np.array([[0.3, 0.7, 1]])
plt.scatter(df.MolLogP, df.Permeability, s=50, c=light_blue, marker='^')
plt.axhline(y=-6, color='magenta', linestyle='--')

plt.show()
