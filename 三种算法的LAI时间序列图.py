import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ==================== 1. 设置文件路径 ====================
# 请根据您的实际文件名修改路径
# 1. 核参数算法结果表（假设列名为 pred_lai）
file1_path = r"E:\叶绿素反演-李文娟老师论文\原始论文中的波段\反演-Result\2026年数据\Final_Inversion_Results_2025.xlsx"
# 2. VP自带/LICI提取的结果表（假设您之前提取为了 Daily_LAI_10_to_14_Median.csv）
file2_path = r"E:\叶绿素反演-李文娟老师论文\VP自带数据\LAI\2026年\LAI.xlsx"
# 3. 9波段算法结果表（假设列名为 pred_lai）
file3_path = r"E:\叶绿素反演-李文娟老师论文\新采用的9波段\反演-Result\Final_Inversion_Results_2026_27Params.xlsx"
# 4. 实测数据表（2026白马LAI.xlsx）
file4_path = r"E:\叶绿素反演-李文娟老师论文\26年白马实测数据\2026白马LAI.xlsx"

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 2. 读取数据 ====================
# 注意：如果是CSV文件请用 read_csv，Excel用 read_excel
df1 = pd.read_excel(file1_path)
df2 = pd.read_excel(file2_path) # 假设这是您刚才生成的CSV
df3 = pd.read_excel(file3_path)
df4 = pd.read_excel(file4_path)

# ==================== 3. 时间序列标准化与清洗 ====================
# 转换日期格式
df1['date'] = pd.to_datetime(df1['date'])
# 如果 file2 的日期列叫 'Date' 或 '日期'，请统一
df2['Date'] = pd.to_datetime(df2['Date'])
df3['date'] = pd.to_datetime(df3['date'])
# 处理实测数据中的纯数字日期格式 (20260319)
df4['日期'] = pd.to_datetime(df4['日期'].astype(str), format='%Y%m%d')

# 强制转换 LAI 相关列为数值，防止字符串干扰
df1['pred_lai'] = pd.to_numeric(df1['pred_lai'], errors='coerce')
df2['LAI_10至14点均值'] = pd.to_numeric(df2['LAI_10至14点均值'], errors='coerce')
df3['pred_lai'] = pd.to_numeric(df3['pred_lai'], errors='coerce')
df4['LAI'] = pd.to_numeric(df4['LAI'], errors='coerce')

# ==================== 4. 设置日期筛选范围 ====================
start_date = '2026-03-20'
end_date = '2026-04-15'
start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)

# 筛选并剔除空值
df1_f = df1[(df1['date'] >= start_dt) & (df1['date'] <= end_dt)].dropna(subset=['pred_lai']).sort_values('date')
df2_f = df2[(df2['Date'] >= start_dt) & (df2['Date'] <= end_dt)].dropna(subset=['LAI_10至14点均值']).sort_values('Date')
df3_f = df3[(df3['date'] >= start_dt) & (df3['date'] <= end_dt)].dropna(subset=['pred_lai']).sort_values('date')
df4_f = df4[(df4['日期'] >= start_dt) & (df4['日期'] <= end_dt)].dropna(subset=['LAI']).sort_values('日期')

# ==================== 5. 绘制时间序列图 ====================
plt.figure(figsize=(11, 7))

# 曲线1：核参数算法 (6波段)
plt.plot(df1_f['date'], df1_f['pred_lai'],
         marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='6波段算法')

# 曲线2：VP自带/LICI算法
plt.plot(df2_f['Date'], df2_f['LAI_10至14点均值'],
         marker='s', linestyle='--', color='#ff7f0e', linewidth=2, label='VP自带LAI数据')

# 曲线3：9波段算法
plt.plot(df3_f['date'], df3_f['pred_lai'],
         marker='^', linestyle='-.', color='#2ca02c', linewidth=2, label='9波段算法')

# ★ 新增：实测数据图层 ★
# 使用红色的巨大五角星表示地面实测值
plt.plot(df4_f['日期'], df4_f['LAI'],
         marker='*', markersize=14, linestyle='', color='red',
         markeredgecolor='black', label='地面实测 (Ground Truth)')

# ==================== 6. 图表装饰与格式化 ====================
# 设置针对 LAI 的 Y 轴刻度 (0-6, 间隔1)
plt.yticks(np.arange(0, 7, 1))
plt.ylim(-0.1, 6.0)

plt.xlabel('日期 (Date)', fontsize=12, fontweight='bold')
plt.ylabel('叶面积指数 LAI ($m^2/m^2$)', fontsize=12, fontweight='bold')
plt.title(f'Multi-Algorithm LAI Time Series Validation\n({start_date} 至 {end_date})', fontsize=14, fontweight='bold')

# 设置X轴日期显示
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.gcf().autofmt_xdate(rotation=45)

# 图例和网格
plt.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('LAI_MultiTable_Comparison.png', dpi=300)
plt.show()