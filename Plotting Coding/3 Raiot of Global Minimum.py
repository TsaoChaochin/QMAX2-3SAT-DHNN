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
# Global Parameters (important!!!)
# =========================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
})
# =========================================================
# Data
# =========================================================
file_path = r"C:\Users\21471\Desktop\Summary1.xlsx"
df1 = pd.read_excel(file_path, sheet_name='Z_G')
# Global minimum ratio
data = df1.iloc[:, 1:].T.to_numpy()

N = np.array([6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120])
cases = ['Case-1', 'Case-2', 'Case-3', 'Case-4', 'Case-5', 'Case-6']

# =========================================================
# Figure
# =========================================================
fig, ax = plt.subplots(figsize=(10, 3.8))
# =========================================================
# Heatmap
# =========================================================
im = ax.imshow(data, aspect='auto', cmap='RdYlBu_r', interpolation='bicubic', vmin=0, vmax=1)

# =========================================================
# Axes
# =========================================================
ax.set_xticks(np.arange(len(N)))
ax.set_xticklabels(N)
ax.set_yticks(np.arange(len(cases)))
ax.set_yticklabels(cases)
ax.set_xlabel(r'$N$')
# ax.set_ylabel('Cases')
# =========================================================
# Minor grid for cell boundaries
# =========================================================
ax.set_xticks(np.arange(-0.5, len(N), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(cases), 1), minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=0.6, alpha=0.55)
ax.tick_params(which='minor', bottom=False, left=False)
# =========================================================
# Remove unnecessary spines
# =========================================================
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
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
            color='black',
            fontsize=7
        )
# =========================================================
# Colorbar
# =========================================================
cbar = fig.colorbar(im, ax=ax, pad=0.01)
cbar.set_label(r'$R_{gm}$', rotation=90)
# =========================================================
# Tight Layout
# =========================================================
plt.tight_layout()
# =========================================================
# Save
# =========================================================
plt.savefig(r'F:\0001Research\01 My Papers\2. Second paper\New Plotting Code\RetrievalPhase\Z_G.pdf', bbox_inches='tight', pad_inches=0)
# plt.savefig(f'{save_dir}\\DHNN_3D_Energy_Surface.png')
plt.show()
