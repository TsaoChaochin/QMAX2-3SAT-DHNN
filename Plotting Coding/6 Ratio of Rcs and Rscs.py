import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
})
# =========================================================
# Data
# =========================================================
file_path = r"C:\Users\21471\Desktop\Summary1.xlsx"
# 'np.mean(Ratio_Solved_Initial)'     'np.mean(Ratio_Solved_Initial_Sa'
df1 = pd.read_excel(file_path, sheet_name='np.mean(Ratio_Solved_Initial)')
# Global minimum ratio
labels = df1.columns[1:]
global_data = df1.iloc[:, 1:].T.to_numpy()
N = np.array([6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120])

# =========================================================
# Colors
# =========================================================
global_color = '#5B8EAD'
local_color = '#E6A57E'
# =========================================================
# Figure
# =========================================================

fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharex=True, sharey=True)
axes = axes.flatten()
# =========================================================
# Plot
# =========================================================
for i, case in enumerate(labels):
    ax = axes[i]
    global_ratio = global_data[i]
    local_ratio = 1 - global_ratio
    # Stacked area
    ax.fill_between(N, 0, global_ratio, color=global_color, alpha=0.92, linewidth=0)
    ax.fill_between(N, global_ratio, 1, color=local_color, alpha=0.92, linewidth=0)
    # Boundary line
    ax.plot(N, global_ratio, color='black', linewidth=1.2)
    # =====================================================
    # Axes style
    # =====================================================
    ax.set_title(case)
    ax.set_xlim(6, 120)
    ax.set_ylim(0, 1)
    ax.set_xticks([6, 30, 60, 90, 120])
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.35)
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
# =========================================================
# Labels
# =========================================================
fig.supxlabel(r'$N$', fontsize=13, y=0.03)
# fig.supylabel('Ratio', fontsize=13, x=0.04)
# =========================================================
# Legend
# =========================================================
handles = [plt.Rectangle((0, 0), 1, 1, color=global_color), plt.Rectangle((0, 0), 1, 1, color=local_color)]

labels = [r'$\overline{R^{c}_{s}}$', r'$1-\overline{R^{c}_{s}}$']
fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.9, 0.02), frameon=False, ncol=2, handlelength=1.6, handletextpad=0.5)
# =========================================================
# Layout
# =========================================================
plt.tight_layout()
# =========================================================
# Save
# =========================================================
plt.savefig(r'F:\0001Research\01 My Papers\2. Second paper\New Plotting Code\RetrievalPhase\MaximumConfiguration\1-Rcs.pdf', dpi=600, bbox_inches='tight')
# plt.savefig('stacked_area_sci.png',  dpi=600, bbox_inches='tight')
plt.show()
