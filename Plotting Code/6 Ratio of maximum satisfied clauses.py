import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

# ========================
# 数据准备
# ========================

N = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]

labels = ['Case-1', 'Case-2', 'Case-3', 'Case-4', 'Case-5', 'Case-6']

file_path = r"C:\Users\21471\Desktop\QMAX2,3SAT\Summary1.xlsx"
df = pd.read_excel(file_path, sheet_name='np.mean(Ratio_Solved_Initial)')

Z = df[['Case-1', 'Case-2', 'Case-3', 'Case-4', 'Case-5', 'Case-6']].values

# X 对应列索引，Y 对应真实 N
X, Y = np.meshgrid(np.arange(len(labels)), N)

# ========================
# 绘图
# ========================
fig = plt.figure(figsize=(7.2, 4.8), constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')

# ========================
# Information Fusion 配色（coolwarm）
# ========================
surf = ax.plot_surface(
    X, Y, Z,
    cmap='coolwarm',
    edgecolor='k',
    linewidth=0.25,
    alpha=0.88
)

ax.scatter(X, Y, Z, s=8, c='black', depthshade=False)

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
ax.set_zlabel(r'$\overline{R^{c}_{s}}$', labelpad=4)

ax.set_xlim(-0.2, len(labels) - 0.8)
ax.set_ylim(min(N), max(N))
ax.set_zlim(np.min(Z), np.max(Z))

# ========================
# 视角
# ========================
ax.view_init(elev=28, azim=-55)
ax.dist = 9.5

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
mappable = cm.ScalarMappable(cmap='coolwarm')
mappable.set_array(Z)

cbar = fig.colorbar(
    mappable,
    ax=ax,
    shrink=0.72,
    aspect=16,
    pad=0.02
)
cbar.set_label(r'$\overline{R^{c}_{s}}$', fontsize=10)

plt.show()

