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
    # font
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    # axis labels (坐标轴标题)
    "axes.labelsize": 13,
    # tick label 刻度字体
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    # legend 字体
    "legend.fontsize": 10,
    # linewidth 线宽
    "lines.linewidth": 1.8,
    # marker size 大小
    "lines.markersize": 5,
    # savefig margin
    "savefig.bbox": "tight",
    # figure dpi（屏幕显示）
    "figure.dpi": 100,
    # save dpi
    "savefig.dpi": 600,
})
# =========================================================
# Data
# =========================================================
# File path
file_path = r"C:\Users\21471\Desktop\Summary1.xlsx"
# 'Z_SAT' 'MAE_SAT' 'F_G'
df1 = pd.read_excel(file_path, sheet_name='F_G')
N = np.array([6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120])
# Read all columns
case1 = df1['Case-1']
case2 = df1['Case-2']
case3 = df1['Case-3']
case4 = df1['Case-4']
case5 = df1['Case-5']
case6 = df1['Case-6']
# =========================================================
# Figure
# constrained_layout 比 tight_layout 更稳定
# =========================================================
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
# =========================================================
# Color-blind Friendly Colors
# =========================================================
colors = ['#2166AC', '#4393C3', '#92C5DE', '#F4A582', '#D6604D', '#B2182B']
markers = ['o', 's', '^', 'D', 'v', 'P']
# =========================================================
# Plot
# =========================================================
ax.plot(N, case1, color=colors[0], marker=markers[0], label='Case-1')
ax.plot(N, case2, color=colors[1], marker=markers[1], label='Case-2')
ax.plot(N, case3, color=colors[2], marker=markers[2], label='Case-3')
ax.plot(N, case4, color=colors[3], marker=markers[3], label='Case-4')
ax.plot(N, case5, color=colors[4], marker=markers[4], label='Case-5')
ax.plot(N, case6, color=colors[5], marker=markers[5], label='Case-6')
# The optimal method is slightly highlighted 最优方法稍微突出一点
# ax.plot(N, case6,color=colors[5], marker=markers[5],linewidth=2.2, label='Case-6')
# =========================================================
# Labels
# =========================================================
ax.set_xlabel(r'$N$')
# 'Z_\mathrm{SAT}' '\mathrm{MAE}_{\mathrm{SAT}}' 'R_{sa}'
ax.set_ylabel(r'$R_{sa}$')
ax.set_xticks(np.arange(6, 121, 6))
# =========================================================
# Grid
# =========================================================
ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.35)
# =========================================================
# Legend
# =========================================================
# lower right    upper right
legend = ax.legend(loc='lower left', ncol=2, frameon=True, fancybox=False, edgecolor='black')
# =========================================================
# Remove unnecessary borders
# =========================================================
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# =========================================================
# Save
# =========================================================
# 'Z_SAT' 'MAE_SAT'
plt.savefig(r'F:\0001Research\01 My Papers\2. Second paper\New Plotting Code\RetrievalPhase\MaximumConfiguration\R_sa.pdf')
# plt.savefig('SCI_Figure.png')
plt.show()
