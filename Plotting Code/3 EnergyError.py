import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

# ========================
# 数据准备
# ========================
N = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60,
     66, 72, 78, 84, 90, 96, 102, 108, 114, 120]

labels = ['Case-1', 'Case-2', 'Case-3', 'Case-4', 'Case-5', 'Case-6']

file_path = r"C:\Users\21471\Desktop\QMAX2,3SAT\Summary1.xlsx"
df = pd.read_excel(file_path, sheet_name='L_G')
Z = df[labels].values

X, Y = np.meshgrid(np.arange(len(labels)), N)

# ========================
# 绘图
# ========================
fig = plt.figure(figsize=(7.2, 4.8), constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')

# ⭐ Elsevier 推荐 colormap
cmap_style = 'GnBu'

surf = ax.plot_surface(
    X, Y, Z,
    cmap=cmap_style,
    edgecolor='gray',
    linewidth=0.25,
    alpha=0.92
)

# 散点改成深灰（不要鲜红）
ax.scatter(X, Y, Z, s=8, c='black', depthshade=False)

# ========================
# 坐标轴设置
# ========================
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=25,
                   ha='right', fontsize=10)

y_ticks = [20, 40, 60, 80, 100, 120]
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks, fontsize=10)

ax.set_ylabel(r'$N$', labelpad=6, fontsize=12)
ax.set_zlabel(r'$L_G$', labelpad=6, fontsize=12)

ax.set_xlim(-0.2, len(labels) - 0.8)
ax.set_ylim(min(N), max(N))
ax.set_zlim(np.min(Z), np.max(Z))

# ========================
# 视角
# ========================
ax.view_init(elev=28, azim=-55)
ax.dist = 9.2

# ========================
# IF 风格弱化背景
# ========================
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis._axinfo["grid"]['linewidth'] = 0.3
ax.yaxis._axinfo["grid"]['linewidth'] = 0.3
ax.zaxis._axinfo["grid"]['linewidth'] = 0.3

# ========================
# 颜色条（字体变大）
# ========================
mappable = cm.ScalarMappable(cmap=cmap_style)
mappable.set_array(Z)

cbar = fig.colorbar(
    mappable,
    ax=ax,
    shrink=0.75,
    aspect=18,
    pad=0.03
)

# ⭐ 字体加大
cbar.set_label(r'$L_G$', fontsize=12)
cbar.ax.tick_params(labelsize=11)

plt.show()
