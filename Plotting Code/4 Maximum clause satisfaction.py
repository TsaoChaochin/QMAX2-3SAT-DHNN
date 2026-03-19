import matplotlib.pyplot as plt
import pandas as pd

N = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]
# File path
file_path = r"C:\Users\21471\Desktop\QMAX2,3SAT\Summary1.xlsx"
df1 = pd.read_excel(file_path, sheet_name='Z_SAT', header=0)  # 第一行作为列名
df1.columns = df1.columns.str.strip()  # 去掉多余空格或隐藏字符
# 取数据列
Z_SAT_MAX2SAT = df1['Case-1']
Z_SAT_MAX2SAT02 = df1['Case-2']
Z_SAT_MAX2SAT04 = df1['Case-3']
Z_SAT_MAX2SAT06 = df1['Case-4']
Z_SAT_MAX2SAT08 = df1['Case-5']
Z_SAT_MAX2SAT10 = df1['Case-6']

# ===== MAE_SAT =====
df2 = pd.read_excel(file_path, sheet_name='MAE_SAT', header=0)
df2.columns = df2.columns.str.strip()
MAE_SAT_MAX2SAT = df2['Case-1']
MAE_SAT_MAX2SAT02 = df2['Case-2']
MAE_SAT_MAX2SAT04 = df2['Case-3']
MAE_SAT_MAX2SAT06 = df2['Case-4']
MAE_SAT_MAX2SAT08 = df2['Case-5']
MAE_SAT_MAX2SAT10 = df2['Case-6']

# ===== F_G =====
df3 = pd.read_excel(file_path, sheet_name='F_G', header=0)
df3.columns = df3.columns.str.strip()
F_G_MAX2SAT = df3['Case-1']
F_G_MAX2SAT02 = df3['Case-2']
F_G_MAX2SAT04 = df3['Case-3']
F_G_MAX2SAT06 = df3['Case-4']
F_G_MAX2SAT08 = df3['Case-5']
F_G_MAX2SAT10 = df3['Case-6']

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10.5,
    "axes.labelsize": 12,
    "axes.linewidth": 1.0,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "mathtext.fontset": "cm",
})

colors = ["#000000", "#4C72B0", "#55A868", "#C44E52", "#8172B2", "#7F7F7F"]


# ===== 绘图函数 =====
def plot_metric(N, metrics, ylabel, legend_loc='upper left', legend_anchor=(0.02, 0.98)):
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)

    labels = ['Case-1', 'Case-2', 'Case-3', 'Case-4', 'Case-5', 'Case-6']
    markers = ['o', 's', '^', 'D', 'v', '>']

    for data, label, color, marker in zip(metrics, labels, colors, markers):
        ax.plot(N, data, color=color, linewidth=1.8, marker=marker, markersize=4.5,
                markerfacecolor='white', markeredgecolor=color,
                markeredgewidth=1.0, label=label)

    ax.set_xlabel(r'$N$', labelpad=6)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', linewidth=0.45, alpha=0.22)
    ax.grid(axis='x', visible=False)
    ax.tick_params(direction='out', length=3.5, width=0.9)
    ax.margins(x=0.015, y=0.04)

    ax.legend(loc=legend_loc,
              bbox_to_anchor=legend_anchor,
              frameon=False, ncol=2,
              handlelength=2.0,
              handletextpad=0.8,
              columnspacing=1.2,
              borderaxespad=0)

    ax.set_xticks(N)
    ax.set_xticklabels([str(n) for n in N], fontsize=9)

    plt.show()


plot_metric(N, [Z_SAT_MAX2SAT, Z_SAT_MAX2SAT02, Z_SAT_MAX2SAT04, Z_SAT_MAX2SAT06, Z_SAT_MAX2SAT08, Z_SAT_MAX2SAT10], r'$Z_{SAT}$')
plot_metric(N, [MAE_SAT_MAX2SAT, MAE_SAT_MAX2SAT02, MAE_SAT_MAX2SAT04, MAE_SAT_MAX2SAT06, MAE_SAT_MAX2SAT08, MAE_SAT_MAX2SAT10], r'$MAE_{SAT}$')
plot_metric(
    N,
    [F_G_MAX2SAT, F_G_MAX2SAT02, F_G_MAX2SAT04,
     F_G_MAX2SAT06, F_G_MAX2SAT08, F_G_MAX2SAT10],
    r'$F_G$',
    legend_loc='lower left',
    legend_anchor=(0.02, 0.02)
)
