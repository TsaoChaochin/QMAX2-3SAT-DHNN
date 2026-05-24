import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.interpolate import griddata
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
    # Font
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    # Axis label size
    "axes.labelsize": 13,
    # Tick label size
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    # Figure DPI
    "figure.dpi": 100,
    # Save DPI
    "savefig.dpi": 600,
    # Tight save
    "savefig.bbox": "tight",
})
# =========================================================
# Data
# =========================================================
N = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]

labels = ['Case-1', 'Case-2', 'Case-3', 'Case-4', 'Case-5', 'Case-6']

file_path = r"C:\Users\21471\Desktop\Summary1.xlsx"
df = pd.read_excel(file_path, sheet_name='np.mean(Ratio_Solved_Initial_Sa')
Z = df[['Case-1', 'Case-2', 'Case-3', 'Case-4', 'Case-5', 'Case-6']].values

# X corresponds to the column index, Y corresponds to the actual N.
X, Y = np.meshgrid(np.arange(len(labels)), N)
# ========================
# Plotting Figures
# ========================
fig = plt.figure(figsize=(9, 6))
plt.tight_layout()
ax = fig.add_subplot(111, projection='3d')
# ========================
# Information Fusion coolwarm viridis cividis
# ========================
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='grey', linewidth=0.15, alpha=0.88)
ax.scatter(X, Y, Z, s=5, c='black', depthshade=False)
# ========================
# 坐标轴设置
# ========================
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
ax.tick_params(axis='x', pad=0)
y_ticks = [20, 40, 60, 80, 100, 120]
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks, fontsize=9)

ax.set_ylabel(r'$N$', labelpad=4)
# \overline{R^{c}_{s}}
ax.set_zlabel(r'$\overline{R^{sc}_{s}}$', labelpad=2)

ax.set_xlim(-0.2, len(labels) - 0.8)
ax.set_ylim(min(N), max(N))
ax.set_zlim(np.min(Z), np.max(Z))

# ========================
# 视角
# ========================
ax.view_init(elev=28, azim=45)

# ========================
# IF 风格：弱化 pane & grid
# ========================
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis._axinfo["grid"]['linewidth'] = 0.3
ax.yaxis._axinfo["grid"]['linewidth'] = 0.3
ax.zaxis._axinfo["grid"]['linewidth'] = 0.3

# ========================
# 颜色条（统一配色）
# ========================
mappable = cm.ScalarMappable(cmap='viridis')
mappable.set_array(Z)

cbar = fig.colorbar(mappable, ax=ax, shrink=0.72, aspect=16, pad=0.07)
# cbar.set_label(r'$\overline{R^{c}_{s}}$', fontsize=10)
plt.savefig(r'F:\0001Research\01 My Papers\2. Second paper\New Plotting Code\RetrievalPhase\MaximumConfiguration\Rscs.pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()
