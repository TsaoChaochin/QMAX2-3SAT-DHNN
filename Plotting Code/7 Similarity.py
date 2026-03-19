import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# ========================
# 数据准备
# ========================

N = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]
labels = ['Case-1', 'Case-2', 'Case-3', 'Case-4', 'Case-5', 'Case-6']

file_path = r"C:\Users\21471\Desktop\QMAX2,3SAT\Summary1.xlsx"
df = pd.read_excel(file_path, sheet_name='np.std(Similarity_list_J_Initia')

Z = df[['Case-1', 'Case-2', 'Case-3', 'Case-4', 'Case-5', 'Case-6']].values

# ========================
# Information Fusion 风格配色
# 蓝 → 灰 → 橙（更高级）
# ========================

cmap = LinearSegmentedColormap.from_list(
    'IF_style',
    ['#3b6ba5', '#d9d9d9', '#d97742'],
    N=256
)

# ========================
# 绘图
# ========================

fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

im = ax.imshow(
    Z,
    aspect='auto',
    cmap=cmap,
    origin='lower'
)

# ========================
# 坐标轴设置
# ========================

ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)

ax.set_yticks(np.arange(len(N)))
ax.set_yticklabels(N, fontsize=9)

ax.set_ylabel(r'$N$', labelpad=4)
ax.set_xlabel('')

# ========================
# 去掉边框（期刊风）
# ========================

for spine in ax.spines.values():
    spine.set_visible(False)

ax.tick_params(axis='both', which='both', length=0)

# ========================
# 颜色条
# ========================

cbar = fig.colorbar(
    im,
    ax=ax,
    shrink=0.85,
    aspect=18,
    pad=0.02
)
cbar.set_label(r'$STD_{S_{jaccard}}$', fontsize=10)
cbar.ax.tick_params(labelsize=9)

plt.show()
