import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ==================== 1. 设置文件路径 ====================
file1_path = r"E:\叶绿素反演-李文娟老师论文\原始论文中的波段\反演-Result\2026年数据\Final_Inversion_Results_2025.xlsx"
file2_path = r"E:\叶绿素反演-李文娟老师论文\原始论文中的波段\白马光谱数据\2026年\利用LICI指数算LCC\反演验证结果_3SV\VP_LCC每日反演数值统计表_2026年.xlsx"
file3_path = r"E:\叶绿素反演-李文娟老师论文\新采用的9波段\反演-Result\Final_Inversion_Results_2026_27Params.xlsx"

# 新增：实测数据的路径（请替换为您的实际路径，如果是CSV格式请用read_csv）
file4_path = r"E:\叶绿素反演-李文娟老师论文\26年白马实测数据\2026白马CAB.xlsx"

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 2. 读取数据 ====================
df1 = pd.read_excel(file1_path)
df2 = pd.read_excel(file2_path)
df3 = pd.read_excel(file3_path)
# 读取实测数据 (根据上传的文件格式，这里假设是CSV)
df4 = pd.read_excel(file4_path)

# ==================== 3. 时间序列标准化 ====================
df1['date'] = pd.to_datetime(df1['date'])
df2['日期'] = pd.to_datetime(df2['日期'])
df3['date'] = pd.to_datetime(df3['date'])

# 新增：处理实测数据中的纯数字日期格式 (例如 20260319)
df4['日期'] = pd.to_datetime(df4['日期'].astype(str), format='%Y%m%d')

# ==================== 4. 设置日期筛选范围 ====================
start_date = '2026-03-20'
end_date = '2026-04-15'

start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)

df1_filtered = df1[(df1['date'] >= start_dt) & (df1['date'] <= end_dt)].sort_values('date')
df2_filtered = df2[(df2['日期'] >= start_dt) & (df2['日期'] <= end_dt)].sort_values('日期')
df3_filtered = df3[(df3['date'] >= start_dt) & (df3['date'] <= end_dt)].sort_values('date')
# 新增：筛选实测数据
df4_filtered = df4[(df4['日期'] >= start_dt) & (df4['日期'] <= end_dt)].sort_values('日期')

# ==================== 5. 绘制时间序列图 ====================
plt.figure(figsize=(10, 6.5))

# 曲线1：核参数算法
plt.plot(df1_filtered['date'], df1_filtered['pred_cab'],
         marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='6波段算法')

# 曲线2：LICI指数算法
plt.plot(df2_filtered['日期'], df2_filtered['估算LCC日均值(μg/cm2)'],
         marker='s', linestyle='--', color='#ff7f0e', linewidth=2, label='LICI指数算法')

# 曲线3：新增的 9波段算法
plt.plot(df3_filtered['date'], df3_filtered['pred_cab'],
         marker='^', linestyle='-.', color='#2ca02c', linewidth=2, label='9波段算法 ')

# ==================== ★ 新增：实测数据图层 ★ ====================
# 使用红色的巨大五角星来表示真实的地面测量值，并且设置 linestyle='' 表示不连线
plt.plot(df4_filtered['日期'], df4_filtered['cab'],
         marker='*', markersize=12, linestyle='', color='red',
         markeredgecolor='black', label='实测数据 (Ground Truth)')

# ==================== 6. 图表装饰与格式化 ====================
# 设置纵坐标刻度
plt.yticks(np.arange(20, 71, 5))

plt.xlabel('日期 (Date)', fontsize=12, fontweight='bold')
plt.ylabel('叶绿素含量 Chlorophyll Content ($\mu g/cm^2$)', fontsize=12, fontweight='bold')
plt.title(f'Chlorophyll Time Series Comparison\n({start_date} to {end_date})', fontsize=14, fontweight='bold')

# 设置X轴日期显示
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate(rotation=45)

# 图例和网格
plt.legend(loc='best', fontsize=10, frameon=True)
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('Time_Series_Comparison_with_GroundTruth.png', dpi=300)
plt.show()