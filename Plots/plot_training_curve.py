import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load space-delimited log file
df = pd.read_csv('pure_log', delim_whitespace=True, header=None, low_memory=False)

# Print and check selected columns
print(df.head())  # Print only the first few rows for quick inspection
# df[10] = pd.to_numeric(df[10], errors='coerce')
print("Train RMS (col 10):", df[10].head())
# print("Validation RMS² (col 14^2):", (df[14]**2).head())

# Plotting
plt.figure(figsize=(10, 6))

# Train RMS
plt.plot(df.index, [float(i[:-1]) for i in df[10]], 'o-', label='Train', color='blue')

# Validation RMS² (optional, uncomment if needed)
plt.plot(df.index, [float(i)**2 for i in df[14]], 's-', label='Validation', color='red')

# Plot formatting
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('L2 Loss', fontsize=20)
plt.title('Training Curve over Epochs', fontsize=20)
plt.tick_params(axis='both', direction='in', labelsize=16)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
plt.tight_layout()
plt.legend(fontsize=20, loc='upper left', bbox_to_anchor=(0.65, 1))
plt.savefig('TrainingCurve.pdf', dpi=300, bbox_inches='tight')
# Show plot
plt.show()
