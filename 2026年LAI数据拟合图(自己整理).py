import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. 读取数据 (建议修改为您本地实际的 Excel/CSV 路径)
# 这里以您提供的格式为例
file_path = r"E:\叶绿素反演-李文娟老师论文\26年白马实测数据\2026白马LAI.xlsx"
# 如果您本地是csv格式，请用 pd.read_csv(file_path)
df = pd.read_excel(file_path)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 数据预处理：强制转换为数值并清洗空值
df['LAI'] = pd.to_numeric(df['LAI'], errors='coerce')
df['6波段'] = pd.to_numeric(df['6波段'], errors='coerce')
df['VP自带'] = pd.to_numeric(df['VP自带'], errors='coerce') # 替换为VP自带
df['9波段'] = pd.to_numeric(df['9波段'], errors='coerce')

# 清洗同时包含实测值和三种算法预测值的有效行
df_clean = df.dropna(subset=['LAI', '6波段', 'VP自带', '9波段'])

# 提取变量：x 为实测值，y 为设备的预测值
x = df_clean['LAI']
y1 = df_clean['6波段']
y2 = df_clean['VP自带']
y3 = df_clean['9波段']

# 2. 绘图设置 (SCI 风格)
# 适当增加高度以容纳包含三个算法的图例
plt.figure(figsize=(8, 6.5))

# --- 算法1：6波段算法 (蓝色圆点, 实线) ---
slope1, intercept1, r_value1, _, _ = stats.linregress(x, y1)
r2_1 = r_value1 ** 2
rmse_1 = np.sqrt(np.mean((x - y1) ** 2))
plt.scatter(x, y1, color='#1f77b4', marker='o', s=50, alpha=0.7)
x_fit = np.linspace(x.min(), x.max(), 100)
plt.plot(x_fit, slope1 * x_fit + intercept1, color='#1f77b4', linestyle='-', linewidth=2,
         label=f'6波段算法 ($R^2={r2_1:.3f}$, $RMSE={rmse_1:.3f}$)')

# --- 算法2：VP自带算法 (红色三角, 虚线) ---
slope2, intercept2, r_value2, _, _ = stats.linregress(x, y2)
r2_2 = r_value2 ** 2
rmse_2 = np.sqrt(np.mean((x - y2) ** 2))
plt.scatter(x, y2, color='#d62728', marker='^', s=50, alpha=0.7)
plt.plot(x_fit, slope2 * x_fit + intercept2, color='#d62728', linestyle='--', linewidth=2,
         label=f'VP自带LAI值 ($R^2={r2_2:.3f}$, $RMSE={rmse_2:.3f}$)')

# --- 算法3：9波段算法 (绿色方块, 点划线) ---
slope3, intercept3, r_value3, _, _ = stats.linregress(x, y3)
r2_3 = r_value3 ** 2
rmse_3 = np.sqrt(np.mean((x - y3) ** 2))
plt.scatter(x, y3, color='#2ca02c', marker='s', s=50, alpha=0.7)
plt.plot(x_fit, slope3 * x_fit + intercept3, color='#2ca02c', linestyle='-.', linewidth=2,
         label=f'9波段算法 ($R^2={r2_3:.3f}$, $RMSE={rmse_3:.3f}$)')

# --- 添加 1:1 参考线 ---
min_val = min(x.min(), y1.min(), y2.min(), y3.min())
max_val = max(x.max(), y1.max(), y2.max(), y3.max())
# 让 1:1 线稍微延伸出数据范围一点
plt.plot([min_val-0.5, max_val+0.5], [min_val-0.5, max_val+0.5], color='gray', linestyle=':', linewidth=1.5, label='1:1 Line')

# 3. 图表细节美化
# ==========================================================
# 核心修改点：强制设定 X 和 Y 轴的显示范围 (LAI 通常在 0-6 左右)
# 您可以根据数据的最大值自行微调为 (0, 5) 或 (0, 8)
plt.xlim(0, 6)
plt.ylim(0, 6)

# 设置刻度密度 (可选，间隔为1)
plt.xticks(np.arange(0, 7, 1))
plt.yticks(np.arange(0, 7, 1))
# ==========================================================

plt.xlabel('Measured LAI ($m^2/m^2$)', fontsize=12, fontweight='bold')
plt.ylabel('Predicted LAI ($m^2/m^2$)', fontsize=12, fontweight='bold')
plt.title('Algorithm Fitting Comparison for Leaf Area Index (LAI)', fontsize=14, fontweight='bold')

# 将图例放在左上角
plt.legend(loc='upper left', fontsize=10, frameon=True)
plt.grid(True, linestyle='--', alpha=0.3)

# 4. 保存与展示
plt.tight_layout()
plt.savefig('fitting_plot_LAI_Algos.png', dpi=300)
plt.show()
