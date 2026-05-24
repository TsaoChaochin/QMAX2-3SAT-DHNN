import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})
# =========================================================
# Read Data
# =========================================================
file_path = r"C:\Users\21471\Desktop\Summary1.xlsx"
df = pd.read_excel(file_path, sheet_name='np.mean(Similarity_list_J_Initi')
N = df['N'].values
cases = ['Case-1', 'Case-2', 'Case-3', 'Case-4', 'Case-5', 'Case-6']
# =========================================================
# Radar Preparation
# =========================================================
num_vars = len(N)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]
# =========================================================
# Figure
# =========================================================
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
# =========================================================
# SCI Soft Colors
# =========================================================
colors = ['#4C72B0', '#DD8452', '#55A868', '#8172B3', '#C44E52', '#64B5CD']
# =========================================================
# Plot each Case
# =========================================================
for i, case in enumerate(cases):
    values = df[case].values.tolist()
    values += values[:1]
    ax.plot(
        angles,
        values,
        linewidth=1.8,
        marker='o',
        markersize=4.5,
        color=colors[i],
        label=case
    )
    ax.fill(angles, values, color=colors[i], alpha=0.08)
# =========================================================
# Theta Labels
# =========================================================
ax.set_xticks(angles[:-1])
ax.set_xticklabels(N)
# =========================================================
# Radial Range
# =========================================================
ax.set_ylim(0.30, 0.41)
# radial tick
ax.tick_params(axis='y', labelsize=9)
# =========================================================
# Grid Style
# =========================================================
ax.grid(linestyle='--', linewidth=0.5, alpha=0.25)
# =========================================================
# Polar Spine
# =========================================================
ax.spines['polar'].set_linewidth(1.0)
# =========================================================
# Radial Tick Position
# =========================================================
ax.set_rlabel_position(0)
# =========================================================
# Legend
# =========================================================
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), ncol=6, frameon=True, fancybox=False, edgecolor='black', fontsize=9, columnspacing=1.1, handlelength=1.6)
# =========================================================
# Save
# =========================================================
plt.savefig(r'F:\0001Research\01 My Papers\2. Second paper\New Plotting Code\RetrievalPhase\MaximumConfiguration\S_J.pdf', bbox_inches='tight', pad_inches=0.0)
plt.show()
