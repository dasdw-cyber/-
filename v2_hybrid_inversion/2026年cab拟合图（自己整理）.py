import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# ==========================================================
# 1. 读取与数据清洗
# ==========================================================
file_path = r"E:\叶绿素反演-李文娟老师论文\26年白马实测数据\2026白马CAB.xlsx"
df = pd.read_excel(file_path)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 算法列表 (按基础到高级的逻辑顺序排列)
algorithms = ['6波段', 'LICI指数算法', '9波段', '6波段+LAI', '9波段+LAI']
colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
markers = ['o', '^', 's', '*', 'D']

# 强制转换并清洗
df['cab'] = pd.to_numeric(df['cab'], errors='coerce')
for algo in algorithms:
    df[algo] = pd.to_numeric(df[algo], errors='coerce')

df_clean = df.dropna(subset=['cab'] + algorithms)
x_true = df_clean['cab']

# ==========================================================
# 2. 创建 2x3 的子图画布 (高清晰度 SCI 风格)
# ==========================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()  # 展平以便遍历

# 统一的坐标轴范围
min_val, max_val = 30, 75
x_fit = np.linspace(min_val, max_val, 100)

# 用于存储指标，以便画最后一张条形图
r2_list = []
rmse_list = []

# ==========================================================
# 3. 循环绘制前 5 个散点拟合图
# ==========================================================
for i, algo in enumerate(algorithms):
    ax = axes[i]
    y_pred = df_clean[algo]

    # 计算统计指标
    slope, intercept, r_value, _, _ = stats.linregress(x_true, y_pred)
    r2 = r_value ** 2
    rmse = np.sqrt(np.mean((x_true - y_pred) ** 2))

    r2_list.append(r2)
    rmse_list.append(rmse)

    # 画散点和拟合线
    ax.scatter(x_true, y_pred, color=colors[i], marker=markers[i], s=60, alpha=0.7, edgecolors='white')
    ax.plot(x_fit, slope * x_fit + intercept, color=colors[i], linestyle='-', linewidth=2)

    # 画 1:1 参考线
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1.5, label='1:1 Line')

    # 美化子图
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_title(algo, fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    # 在图内部添加文本框显示 R2 和 RMSE
    text_str = f"$R^2$ = {r2:.3f}\n$RMSE$ = {rmse:.3f}"
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # 仅在左侧和底部显示标签，避免重复冗余
    if i >= 3:
        ax.set_xlabel('Measured $C_{ab}$ ($\mu g/cm^2$)', fontsize=11)
    if i % 3 == 0:
        ax.set_ylabel('Predicted $C_{ab}$ ($\mu g/cm^2$)', fontsize=11)

# ==========================================================
# 4. 在第 6 个子图绘制综合精度柱状图
# ==========================================================
ax_bar = axes[5]
x_pos = np.arange(len(algorithms))
width = 0.35

# 绘制双轴柱状图 (左轴 RMSE, 右轴 R2)
ax_r2 = ax_bar.twinx()

rects1 = ax_bar.bar(x_pos - width / 2, rmse_list, width, label='RMSE (越低越好)', color='#d62728', alpha=0.7)
rects2 = ax_r2.bar(x_pos + width / 2, r2_list, width, label='$R^2$ (越高越好)', color='#1f77b4', alpha=0.7)

ax_bar.set_ylabel('RMSE', fontsize=11, color='#d62728', fontweight='bold')
ax_r2.set_ylabel('$R^2$', fontsize=11, color='#1f77b4', fontweight='bold')
ax_bar.set_title('各算法综合精度排名', fontsize=13, fontweight='bold', pad=10)

ax_bar.set_xticks(x_pos)
# 为了防止文字太长重叠，对长名字进行换行
short_names = [name.replace('+LAI', '\n+LAI').replace('算法', '') for name in algorithms]
ax_bar.set_xticklabels(short_names, fontsize=10, rotation=0)

# 合并图例
lines1, labels1 = ax_bar.get_legend_handles_labels()
lines2, labels2 = ax_r2.get_legend_handles_labels()
ax_bar.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

ax_bar.grid(axis='y', linestyle='--', alpha=0.4)

# ==========================================================
# 5. 全局排版与保存
# ==========================================================
plt.tight_layout()
# 微调高度以留出顶部的主标题空间
fig.subplots_adjust(top=0.92)
fig.suptitle('Multi-Algorithm Evaluation for Chlorophyll ($C_{ab}$) Inversion', fontsize=16, fontweight='bold')

output_img = 'Subplot_5_Algos_Comprehensive.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"✅ 图表绘制完成！已保存为高清图片: {output_img}")
plt.show()