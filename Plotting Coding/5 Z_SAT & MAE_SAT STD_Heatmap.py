import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# =========================================================
# Optional: SCI style
# =========================================================
try:
    import scienceplots
    plt.style.use(['science'])
except:
    plt.style.use('default')
# =========================================================
# Global Parameters
# =========================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 100,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# =========================================================
# Read Data
# =========================================================
file_path = r"C:\Users\21471\Desktop\Summary1.xlsx"
# 'STD_{Z_SAT}'  'STD_{MAE_SAT}'
df = pd.read_excel(file_path, sheet_name='STD_{MAE_SAT}')
# =========================================================
# Construct Matrix
# =========================================================
data = np.array([df['Case-1'], df['Case-2'], df['Case-3'], df['Case-4'], df['Case-5'], df['Case-6']])
# =========================================================
# Figure
# =========================================================
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
# =========================================================
# Heatmap
# =========================================================
im = ax.imshow(data, aspect='auto', cmap='magma', interpolation='nearest')
# =========================================================
# Axis Labels
# =========================================================
ax.set_xlabel(r'$N$')
# ax.set_ylabel('Cases')
# X-axis
N = np.array([6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120])
ax.set_xticks(np.arange(len(N)))
ax.set_xticklabels(N)
# ax.set_xticks(np.arange(0, len(N), 2))
# ax.set_xticklabels(N[::2])
# Y-axis
ax.set_yticks(np.arange(6))
ax.set_yticklabels(['Case-1', 'Case-2', 'Case-3', 'Case-4', 'Case-5', 'Case-6'])
# =========================================================
# Add Value Annotation
# =========================================================
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(
            j, i,
            f'{data[i, j]:.2f}',
            ha='center',
            va='center',
            color='white',
            fontsize=7
        )
# =========================================================
# Colorbar
# =========================================================
cbar = plt.colorbar(im, ax=ax, pad=0.01)
# '\mathrm{STD}_{Z_\mathrm{SAT}}' '\mathrm{STD}_{\mathrm{MAE}_\mathrm{SAT}}'
cbar.set_label(r'$\mathrm{STD}_{\mathrm{MAE}_\mathrm{SAT}}$')
# =========================================================
# Remove Borders
# =========================================================
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# =========================================================
# Save
# =========================================================
# 'STD_{MAE_Learn}' 'STD_{RMSE_Learn}' 'STD_{MAPE_Learn}'
plt.savefig(r'F:\0001Research\01 My Papers\2. Second paper\New Plotting Code\RetrievalPhase\MaximumConfiguration\STD_MAE_SAT_Heatmap.pdf')

plt.show()