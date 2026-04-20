import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ==========================================================
# 1. 读取与数据清洗
# ==========================================================
# 请替换为您本地实际的文件路径
file_path = r"E:\叶绿素反演-李文娟老师论文\引入LAI作为先验知识输入\实测cab\2026白马cab-2.xlsx"
df = pd.read_excel(file_path)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 定义要对比的两列算法名 (必须和表头完全一致)
algo_6band = '6波段+反演LAI一起作为输入'
algo_9band = '9波段+反演LAI一起作为输入'

# 强制转换并清洗空值
df['cab'] = pd.to_numeric(df['cab'], errors='coerce')
df[algo_6band] = pd.to_numeric(df[algo_6band], errors='coerce')
df[algo_9band] = pd.to_numeric(df[algo_9band], errors='coerce')

df_clean = df.dropna(subset=['cab', algo_6band, algo_9band])

x_true = df_clean['cab']
y_6band = df_clean[algo_6band]
y_9band = df_clean[algo_9band]

# ==========================================================
# 2. 绘图设置 (高清晰度 SCI 风格)
# ==========================================================
plt.figure(figsize=(8, 6.5))
x_fit = np.linspace(30, 75, 100)

# --- 绘制算法 1：6波段+预测LAI串联反演 (蓝色圆点) ---
slope6, intercept6, r_value6, _, _ = stats.linregress(x_true, y_6band)
r2_6 = r_value6 ** 2
rmse_6 = np.sqrt(np.mean((x_true - y_6band) ** 2))

plt.scatter(x_true, y_6band, color='#1f77b4', marker='o', s=70, alpha=0.8, edgecolors='white', zorder=3)
plt.plot(x_fit, slope6 * x_fit + intercept6, color='#1f77b4', linestyle='-', linewidth=2.5,
         label=f'6波段 (融合反演LAI)  [$R^2={r2_6:.3f}$, $RMSE={rmse_6:.3f}$]', zorder=2)

# --- 绘制算法 2：9波段+预测LAI串联反演 (红色方形) ---
slope9, intercept9, r_value9, _, _ = stats.linregress(x_true, y_9band)
r2_9 = r_value9 ** 2
rmse_9 = np.sqrt(np.mean((x_true - y_9band) ** 2))

plt.scatter(x_true, y_9band, color='#d62728', marker='s', s=70, alpha=0.8, edgecolors='white', zorder=4)
plt.plot(x_fit, slope9 * x_fit + intercept9, color='#d62728', linestyle='--', linewidth=2.5,
         label=f'9波段 (融合反演LAI)  [$R^2={r2_9:.3f}$, $RMSE={rmse_9:.3f}$]', zorder=2)

# --- 添加 1:1 参考线 ---
plt.plot([30, 75], [30, 75], color='gray', linestyle=':', linewidth=2, label='1:1 Line', zorder=1)

# ==========================================================
# 3. 细节美化与标签
# ==========================================================
plt.xlim(30, 70)
plt.ylim(30, 70)

plt.xlabel('Measured $C_{ab}$ ($\mu g/cm^2$)', fontsize=13, fontweight='bold')
plt.ylabel('Predicted $C_{ab}$ ($\mu g/cm^2$)', fontsize=13, fontweight='bold')
plt.title('Cascaded Inversion Evaluation: 6-Band vs 9-Band', fontsize=15, fontweight='bold', pad=15)

# 图例放置在右下角，使用半透明底色防止遮盖数据点
plt.legend(loc='lower right', fontsize=11, frameon=True, facecolor='white', framealpha=0.9, edgecolor='gray')
plt.grid(True, linestyle='--', alpha=0.4)

# ==========================================================
# 4. 保存与展示
# ==========================================================
plt.tight_layout()
output_filename = 'Cascaded_Inversion_Comparison_6vs9.png'
plt.savefig(output_filename, dpi=300)
print(f"✅ 图表绘制完成！已保存为高清图片: {output_filename}")
plt.show()