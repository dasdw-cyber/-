import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import warnings

# === 忽略运行时的警告 ===
warnings.filterwarnings('ignore')

# === 解决 matplotlib 中文显示问题 ===
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 🎯 1. 参数配置区
# ==========================================
# 指向经过 NDVI 筛选后保存的总数据表
# 注意：请根据你的实际盘符和路径进行修改
data_path = Path(r"E:\叶绿素反演-李文娟老师论文\白马光谱数据\NDVI_数据筛选结果\错误辐射定标\Abnormal_Data_综合筛选.csv")

# 填写你想批量绘图的特定月份 (格式 'YYYY-MM')
target_month = '2025-03'

# 图片输出路径（在数据表同目录下创建一个“光谱曲线”文件夹）
output_dir = data_path.parent / "光谱曲线"
output_dir.mkdir(parents=True, exist_ok=True)  # 如果文件夹不存在则自动创建
output_img_path = output_dir / f"{target_month}_10至14点_每日平均反射率光谱曲线.png"
# ==========================================
# ⚙️ 2. 数据加载与时间解析
# ==========================================
print(f"📥 正在读取数据: {data_path.name} ...")
if not data_path.exists():
    raise FileNotFoundError(f"找不到文件，请检查路径是否正确: {data_path}")

df = pd.read_csv(data_path)

# 解析时间维度
df['时间'] = pd.to_datetime(df['时间'])
df['解析日期'] = df['时间'].dt.strftime('%Y-%m-%d')
df['解析年月'] = df['时间'].dt.strftime('%Y-%m')  # 提取年月用于匹配
df['解析时间_小时小数'] = df['时间'].dt.hour + df['时间'].dt.minute / 60.0

# ==========================================
# 🔍 3. 筛选指定月份并获取该月所有可用日期
# ==========================================
month_df = df[df['解析年月'] == target_month]
if month_df.empty:
    raise ValueError(f"❌ 数据表中没有找到 {target_month} 的数据，请检查月份设置！")

# 获取该月中所有不重复的日期，并进行排序
unique_dates = sorted(month_df['解析日期'].unique())
print(f"📅 在 {target_month} 中共检测到 {len(unique_dates)} 天包含数据。")

# ==========================================
# 🔍 4. 精准提取并排序反射率波段
# ==========================================
ref_cols = [c for c in df.columns if str(c).startswith('Reflectance-')]
if not ref_cols:
    raise ValueError("数据表中未找到 'Reflectance-' 开头的列，请确认输入数据无误！")

# 提取波长数值并进行升序排序，确保画图时横坐标连线不混乱
wavelengths = [float(c.split('-')[1]) for c in ref_cols]
sorted_pairs = sorted(zip(wavelengths, ref_cols))
sorted_wavelengths = [pair[0] for pair in sorted_pairs]
sorted_ref_cols = [pair[1] for pair in sorted_pairs]

# ==========================================
# 📈 5. 筛选并绘制曲线 (引入时间渐变色)
# ==========================================
print("🎨 开始计算 10:00-14:00 区间的平均反射率并绘图...")
plt.figure(figsize=(12, 7), dpi=150)  # 稍微加宽画布以容纳多出来的图例
plot_count = 0

# 生成随日期渐变的颜色组 (viridis 是经典的科学制图渐变色，深紫->蓝绿->明黄)
# 这样画出来的线能直观地表现出随时间推移的光谱变化趋势
colors = cm.viridis(np.linspace(0, 1, len(unique_dates)))

for i, date_str in enumerate(unique_dates):
    # 逻辑：匹配具体日期 且 时间在 10.0 到 14.0 之间
    mask = (df['解析日期'] == date_str) & (df['解析时间_小时小数'] >= 10.0) & (df['解析时间_小时小数'] <= 14.0)
    sub_df = df[mask]

    if not sub_df.empty:
        # 提取排好序的波段列，计算该时间段内的均值
        mean_reflectance = sub_df[sorted_ref_cols].mean().values

        # 绘制光谱曲线，按时间顺序赋予渐变色
        plt.plot(sorted_wavelengths, mean_reflectance, color=colors[i], marker='.',
                 markersize=3, linewidth=1.5, label=f'{date_str}')
        plot_count += 1
    else:
        print(f"⚠️ 提示: {date_str} 没有 10:00-14:00 之间的数据，跳过该日期。")

# ==========================================
# 🖼️ 6. 图表美化与输出
# ==========================================
if plot_count > 0:
    plt.title(f"{target_month} 每日冠层平均反射率光谱曲线 (10:00 - 14:00)", fontsize=16, pad=15)
    plt.xlabel("波长 Wavelength (nm)", fontsize=13)
    plt.ylabel("反射率 Reflectance", fontsize=13)

    # 针对叶绿素反演的关键光谱区间进行高亮标注
    plt.axvspan(520, 580, color='green', alpha=0.1, label='绿峰 (Green Peak ~550nm)')
    plt.axvspan(650, 680, color='blue', alpha=0.05, label='红谷 (Red Well ~660nm)')
    plt.axvspan(680, 750, color='red', alpha=0.1, label='红边 (Red Edge 680-750nm)')

    # 设置 y 轴底线为 0
    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 图例智能化排版：如果天数超过15天，自动将图例拆分为多列，防止遮挡或超出屏幕
    ncol = 1 if plot_count <= 15 else (2 if plot_count <= 30 else 3)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=ncol, title="日期 (Date)", fontsize=9)

    plt.tight_layout()

    # 保存与显示
    plt.savefig(output_img_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n✅ 光谱曲线生成成功！共绘制了 {plot_count} 天的有效曲线。图片已保存至:\n   {output_img_path}")
else:
    print("\n❌ 绘图失败：该月份所有检测到的日期均无满足条件 (10:00-14:00) 的数据。")