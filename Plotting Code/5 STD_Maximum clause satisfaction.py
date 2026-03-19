import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

# ========================
# 数据准备
# ========================
N = np.array([6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120])
labels = ['Case-1', 'Case-2', 'Case-3', 'Case-4', 'Case-5', 'Case-6']

file_path = r"C:\Users\21471\Desktop\QMAX2,3SAT\Summary1.xlsx"
df = pd.read_excel(file_path, sheet_name='STD_{MAE_SAT}')
Z = df[['Case-1', 'Case-2', 'Case-3', 'Case-4', 'Case-5', 'Case-6']].values

# ========================
# bar3d 坐标（meshgrid + flatten）
# ========================
xpos, ypos = np.meshgrid(np.arange(len(labels)), np.arange(len(N)))
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)
dz = Z.flatten()

# ========================
# 柱宽
# ========================
dx = dy = 0.42

# ========================
# 颜色（Information Fusion 风格渐变）
# ========================
norm = Normalize(np.min(Z), np.max(Z))
colors_if = LinearSegmentedColormap.from_list(
    'deep_ocean',
    ['#0a192f', '#112240', '#1f4068', '#1f77b4', '#4cc9f0'],
    N=256
)

colors = colors_if(norm(dz))  # 每个柱子颜色根据高度

# ========================
# 绘图
# ========================
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')
ax.set_position([0.08, 0.12, 0.72, 0.80])  # 左边压缩，给 colorbar 留空间

ax.bar3d(
    xpos, ypos, zpos,
    dx, dy, dz,
    color=colors,
    edgecolor='0.35',
    linewidth=0.25,
    alpha=0.88,
    shade=False
)

# ========================
# 坐标轴设置
# ========================
ax.set_xticks(np.arange(len(labels)) + dx / 2)
ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)

# y轴刻度修改（显示 20,40,...,120）
yticks_labels = [20, 40, 60, 80, 100, 120]  # 显示刻度标签
yticks_pos = np.array([3, 6, 9, 12, 15, 18]) + dy / 2  # 对应 N 的索引位置
ax.set_yticks(yticks_pos)
ax.set_yticklabels(yticks_labels, fontsize=9)

ax.set_ylabel(r'$N$', labelpad=4)
ax.set_zlabel(r'$STD_{MAE_{SAT}}$', labelpad=4)

ax.set_xlim(-0.2, len(labels))
ax.set_ylim(-0.2, len(N))
ax.set_zlim(0, dz.max() * 1.08)

# ========================
# 视角
# ========================
ax.view_init(elev=24, azim=-58)
ax.dist = 9

# ========================
# 弱化 pane & grid
# ========================
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis._axinfo["grid"]['linewidth'] = 0.3
ax.yaxis._axinfo["grid"]['linewidth'] = 0.3
ax.zaxis._axinfo["grid"]['linewidth'] = 0.3

# ========================
# 右侧颜色条
# ========================
mappable = ScalarMappable(cmap=colors_if, norm=norm)
mappable.set_array(Z)
cbar = fig.colorbar(
    mappable,
    ax=ax,
    shrink=0.7,
    aspect=16,
    pad=0.06
)
cbar.set_label(r'$STD_{MAE_{SAT}}$', fontsize=10)
cbar.outline.set_linewidth(0.6)
cbar.ax.tick_params(labelsize=9)
plt.show()
