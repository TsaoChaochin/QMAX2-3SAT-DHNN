import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

# =========================================================
# Optional SCI style
# =========================================================
try:
    import scienceplots

    plt.style.use(['science'])
except:
    plt.style.use('default')
# =========================================================
# Global Style
# =========================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 100,
    "savefig.dpi": 600,
})
# =========================================================
# Read Data
# =========================================================
file_path = r"C:\Users\21471\Desktop\Summary1.xlsx"
df = pd.read_excel(file_path, sheet_name='np.std(Similarity_list_J_Initia')
# =========================================================
# Data
# =========================================================
N = df['N'].values
cases = [
    'Case-6',
    'Case-5',
    'Case-4',
    'Case-3',
    'Case-2',
    'Case-1'
]
data = df[cases].values.T
# =========================================================
# Polar Grid
# =========================================================
num_theta = len(N)
num_r = len(cases)
theta = np.linspace(0, 2 * np.pi, num_theta + 1)
# ring positions
r = np.arange(num_r + 1)
# meshgrid
Theta, R = np.meshgrid(theta, r)
# =========================================================
# Figure
# =========================================================
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
# =========================================================
# Colormap
# =========================================================
cmap = plt.cm.inferno
# cmap = plt.cm.magma
# cmap = plt.cm.cividis
# norm = LogNorm(vmin=data.min(), vmax=data.max())
# norm = PowerNorm(gamma=0.55, vmin=data.min(), vmax=data.max())
norm = Normalize(vmin=0.004, vmax=0.08)
# norm = Normalize(vmin=data.min(), vmax=data.max())
# =========================================================
# Draw Circular Heatmap
# =========================================================
pcm = ax.pcolormesh(Theta, R, data, cmap=cmap, norm=norm, shading='flat', edgecolors='black', linewidth=1.2)
# =========================================================
# Theta Labels
# =========================================================
ax.set_xticks(theta[:-1])
ax.set_xticklabels(N)
# =========================================================
# Radial Labels
# =========================================================
ax.set_yticks(np.arange(num_r) + 0.5)
ax.set_yticklabels(cases, color='white')
# =========================================================
# Style
# =========================================================
ax.grid(False)
ax.spines['polar'].set_visible(False)
ax.set_rlabel_position(135)
# =========================================================
# Colorbar
# =========================================================
cbar = plt.colorbar(pcm, pad=0.04, shrink=0.85)
# cbar.set_label('Standard Deviation', rotation=270, labelpad=18)
# fig.patch.set_facecolor('#F8F8F8')
# ax.set_facecolor('#F8F8F8')
# =========================================================
# Layout
# =========================================================
plt.tight_layout()
# =========================================================
# Save
# =========================================================
plt.savefig(r'F:\0001Research\01 My Papers\2. Second paper\New Plotting Code\RetrievalPhase\MaximumConfiguration\STD_S_J.pdf', bbox_inches='tight', pad_inches=0.0)
plt.show()
