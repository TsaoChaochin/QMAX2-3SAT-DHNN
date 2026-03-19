import matplotlib.pyplot as plt
import pandas as pd

N = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]
# File path
file_path = r"C:\Users\21471\Desktop\Summary1.xlsx"

xls = pd.ExcelFile(file_path)
# print("Name of worksheets：", xls.sheet_names
df1 = pd.read_excel(file_path, sheet_name='MAE_Learn')
# Read all columns
MAX2SAT = df1['Case-1']
MAX2SAT02 = df1['Case-2']
MAX2SAT04 = df1['Case-3']
MAX2SAT06 = df1['Case-4']
MAX2SAT08 = df1['Case-5']
MAX2SAT10 = df1['Case-6']

# ===== RMSE =====
df2 = pd.read_excel(file_path, sheet_name='RMSE_Learn')
RMSE_MAX2SAT = df2['Case-1']
RMSE_MAX2SAT02 = df2['Case-2']
RMSE_MAX2SAT04 = df2['Case-3']
RMSE_MAX2SAT06 = df2['Case-4']
RMSE_MAX2SAT08 = df2['Case-5']
RMSE_MAX2SAT10 = df2['Case-6']

# ===== MAPE =====
df3 = pd.read_excel(file_path, sheet_name='MAPE_Learn')
MAPE_MAX2SAT = df3['Case-1']
MAPE_MAX2SAT02 = df3['Case-2']
MAPE_MAX2SAT04 = df3['Case-3']
MAPE_MAX2SAT06 = df3['Case-4']
MAPE_MAX2SAT08 = df3['Case-5']
MAPE_MAX2SAT10 = df3['Case-6']

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
def plot_metric(N, metrics, ylabel, legend_loc='upper left'):
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

    # ⭐ 修改这里
    ax.legend(loc=legend_loc,
              frameon=False, ncol=2,
              handlelength=2.0,
              handletextpad=0.8,
              columnspacing=1.2)

    ax.set_xticks(N)
    ax.set_xticklabels([str(n) for n in N], fontsize=9)

    plt.show()


plot_metric(N,
            [MAX2SAT, MAX2SAT02, MAX2SAT04, MAX2SAT06, MAX2SAT08, MAX2SAT10],
            r'$MAE_{\mathrm{Learn}}$')

plot_metric(N,
            [RMSE_MAX2SAT, RMSE_MAX2SAT02, RMSE_MAX2SAT04, RMSE_MAX2SAT06, RMSE_MAX2SAT08, RMSE_MAX2SAT10],
            r'$RMSE_{\mathrm{Learn}}$')

# ⭐ 这里改为右下角
plot_metric(N,
            [MAPE_MAX2SAT, MAPE_MAX2SAT02, MAPE_MAX2SAT04, MAPE_MAX2SAT06, MAPE_MAX2SAT08, MAPE_MAX2SAT10],
            r'$MAPE_{\mathrm{Learn}}$',
            legend_loc='lower right')
