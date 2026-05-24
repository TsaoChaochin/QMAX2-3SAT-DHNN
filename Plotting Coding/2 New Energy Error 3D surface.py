import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =========================================================
# Optional: SCI Style
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
file_path = r"C:\Users\21471\Desktop\Summary1.xlsx"
df1 = pd.read_excel(file_path, sheet_name='MAE_Energy')
Z = df1.iloc[:, 1:].T.to_numpy()

N = np.array([6, 12, 18, 24, 30, 6, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120])
cases = np.array([1, 2, 3, 4, 5, 6])
# =========================================================
# Meshgrid
# =========================================================
X, Y = np.meshgrid(N, cases)
# =========================================================
# Figure
# =========================================================
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
# =========================================================
# Surface Plot
# =========================================================
surf = ax.plot_surface(X, Y, Z, cmap='magma', edgecolor='none', antialiased=True, alpha=0.95)
# =========================================================
# Bottom Contour Projection
# Highly recommended, greatly improves SCI style
# =========================================================
ax.contourf(X, Y, Z, zdir='z', offset=0, cmap='magma', levels=30, alpha=0.85)
# =========================================================
# Labels
# =========================================================
ax.set_xlabel(r'$N$', labelpad=10)
ax.set_ylabel('Cases', labelpad=10)
ax.set_zlabel(r'$\mathrm{MAE}_\mathrm{Energy}$', labelpad=0)
# =========================================================
# Axis Ticks
# =========================================================
ax.set_yticks(cases)
ax.set_yticklabels(['C1', 'C2', 'C3', 'C4', 'C5', 'C6'])
# =========================================================
# View Angle
# 这个角度很关键
# =========================================================
ax.view_init(elev=28, azim=-130)
# =========================================================
# Remove pane fills (A stronger sense of sophistication 高级感更强)
# =========================================================
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
# =========================================================
# Grid Style
# =========================================================
ax.grid(True, linestyle='--', alpha=0.3)
# =========================================================
# Colorbar
# =========================================================
cbar = fig.colorbar(surf, shrink=0.72, aspect=18, pad=0.0)
# '$\mathrm{MAE}_{\mathrm{Learn}}$'
# cbar.set_label(r'$\mathrm{MAE}_\mathrm{Energy}$', fontsize=11)
# =========================================================
# Aspect Ratio
# Very important
# =========================================================
ax.set_box_aspect((2.4, 1.2, 1.4))
# =========================================================
# Save
# =========================================================
# save_dir = r'/LearningPhase'
plt.savefig(r'F:\0001Research\01 My Papers\2. Second paper\New Plotting Code\RetrievalPhase\MAE_Energy.pdf', bbox_inches='tight', pad_inches=0.2)
# plt.savefig(f'{save_dir}\\DHNN_3D_Energy_Surface.png')
# =========================================================
# Show
# =========================================================
plt.show()
