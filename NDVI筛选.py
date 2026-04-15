import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import warnings

# === 忽略运行时的警告 ===
warnings.filterwarnings('ignore')

# === 解决 matplotlib 中文显示问题 ===
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 支持中文字体
plt.rcParams['axes.unicode_minus'] = False

# === 1. 基础设置与路径 ===
# 读取上一个流式定标脚本输出的根目录
input_dir = Path(r"E:\叶绿素反演-李文娟老师论文\原始论文中的波段\白马光谱数据\2026年\辐射定标数据")
# 本次 NDVI 筛选和报告存放的根目录
save_dir = Path(r"E:\叶绿素反演-李文娟老师论文\原始论文中的波段\白马光谱数据\2026年\NDVI_数据筛选结果")
save_dir.mkdir(exist_ok=True, parents=True)

pdf_path = save_dir / "NDVI异常溯源报告.pdf"

print("正在读取所有辐射定标数据...")
# 自动搜索所有年月文件夹下的 csv 文件 (排除 Unknown_Time 文件夹)
all_csv_files = [f for f in input_dir.rglob("*.csv") if "Unknown_Time" not in str(f)]

if not all_csv_files:
    raise FileNotFoundError(f"在 {input_dir} 中没有找到任何 CSV 数据，请检查路径！")

# 合并所有数据
df_list = [pd.read_csv(f) for f in all_csv_files]
df = pd.concat(df_list, ignore_index=True)

# 处理时间列 —— 【已修复：加入 format='mixed' 解决格式不一致的报错】
df['时间'] = pd.to_datetime(df['时间'], format='mixed')
df['日期'] = df['时间'].dt.date
df['时间数值'] = df['时间'].dt.hour + df['时间'].dt.minute / 60


# === 2. 自动寻找波段列 ===
def find_closest_column(df_data, target_wavelength, prefix="Reflectance-"):
    cols = [c for c in df_data.columns if str(c).startswith(prefix)]
    if not cols:
        raise ValueError(f"找不到以 {prefix} 开头的列，请确认上一部辐射定标是否输出了该数据！")
    wavelengths = [float(c.split('-')[1]) for c in cols]
    idx = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i] - target_wavelength))
    return cols[idx], wavelengths[idx]


red_col, red_wv = find_closest_column(df, 660, "Reflectance-")
nir_col, nir_wv = find_closest_column(df, 800, "Reflectance-")
print(f"✅ 使用 RED 波段: {red_col} ({red_wv} nm), NIR 波段: {nir_col} ({nir_wv} nm)")

E_up_red, _ = find_closest_column(df, red_wv, "E_up-")
E_up_nir, _ = find_closest_column(df, nir_wv, "E_up-")
E_down_red, _ = find_closest_column(df, red_wv, "E_down-")
E_down_nir, _ = find_closest_column(df, nir_wv, "E_down-")

print(f"☀️ 辐射分析 RED 波段: {E_up_red} / {E_down_red}")
print(f"☀️ 辐射分析 NIR 波段: {E_up_nir} / {E_down_nir}")

# === 3. 计算 NDVI ===
print("正在计算 NDVI...")
df['NDVI'] = (df[nir_col] - df[red_col]) / (df[nir_col] + df[red_col])

# === 4. 数据统计与异常识别 (按天) ===
# 计算每天的 mean 和 std
daily_stats = df.groupby('日期')['NDVI'].agg(['mean', 'std'])

# 设定判断阈值
threshold_std = 0.1
threshold_mean = 0.15

# 依据新规则：正常数据为 std < 0.1 & mean > 0.15；反之则为异常
normal_dates = daily_stats[(daily_stats['std'] < threshold_std) & (daily_stats['mean'] > threshold_mean)].index
abnormal_dates = daily_stats[~daily_stats.index.isin(normal_dates)].index

print(f"\n📊 统计结果：共计 {len(daily_stats)} 天数据")
print(f"   🟢 正常天数: {len(normal_dates)} 天")
print(f"   🚨 异常天数: {len(abnormal_dates)} 天")

# === 5. 数据划分与保存 ===
normal_df = df[df['日期'].isin(normal_dates)]
abnormal_df = df[df['日期'].isin(abnormal_dates)]

# 为总表打上标签，方便后续建模时筛选
df['数据状态'] = '异常'
df.loc[df['日期'].isin(normal_dates), '数据状态'] = '正常'

# 设置保存路径
all_save_path = save_dir / "All_Data_总数据表.csv"
normal_save_path = save_dir / "Normal_Data_正常数据.csv"
abnormal_save_path = save_dir / "Abnormal_Data_异常数据.csv"

# 剔除辅助计算的列，保存整洁的 CSV
cols_to_drop = ['时间数值', '日期']

# 保存总数据表
df.drop(columns=cols_to_drop).to_csv(all_save_path, index=False, encoding='utf-8-sig')
print(f"💾 总数据表已保存至: {all_save_path}")

# 保存正常与异常拆分表
normal_df.drop(columns=cols_to_drop).to_csv(normal_save_path, index=False, encoding='utf-8-sig')
abnormal_df.drop(columns=cols_to_drop).to_csv(abnormal_save_path, index=False, encoding='utf-8-sig')
print(f"💾 正常数据已保存至: {normal_save_path}")
print(f"💾 异常数据已保存至: {abnormal_save_path}")

# === 6. 生成异常日期的 PDF 诊断报告 ===
if len(abnormal_dates) > 0:
    print(f"\n正在生成异常溯源 PDF 报告，请稍候...")
    with PdfPages(pdf_path) as pdf:
        # --- 报告封面 ---
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')

        text = (f"📄 NDVI 异常溯源报告\n\n"
                f"【异常判定标准】：\n"
                f"  不满足 (NDVI标准差 < {threshold_std} 且 NDVI均值 > {threshold_mean}) 的所有日期\n\n"
                f"【统计概况】：\n"
                f"  总天数：{len(daily_stats)} 天\n"
                f"  异常天数：{len(abnormal_dates)} 天\n\n"
                f"【异常日期列表】：\n")

        # 只列出前 30 个避免封面溢出
        for i, d in enumerate(abnormal_dates[:30]):
            std_val = daily_stats.loc[d, 'std']
            mean_val = daily_stats.loc[d, 'mean']
            text += f"  - {d} | std={std_val:.3f}, mean={mean_val:.3f}\n"
        if len(abnormal_dates) > 30:
            text += f"  - ... 等共 {len(abnormal_dates)} 天\n"

        plt.text(0.05, 0.95, text, va='top', fontsize=12, family='monospace')
        pdf.savefig()
        plt.close()

        # --- 为每个异常日期绘制诊断图 ---
        for d in abnormal_dates:
            sub = df[df['日期'] == d].sort_values('时间数值')

            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # 上图: 反射率 + NDVI（点+线）
            axes[0].plot(sub['时间数值'], sub[red_col], marker='o', markersize=5, label=f'Red {red_wv:.1f}nm', color='red')
            axes[0].plot(sub['时间数值'], sub[nir_col], marker='o', markersize=5, label=f'NIR {nir_wv:.1f}nm', color='green')
            axes[0].plot(sub['时间数值'], sub['NDVI'], marker='o', markersize=5, label='NDVI', color='blue', linestyle='--')
            axes[0].set_ylim(-1, 1)
            axes[0].set_ylabel("Reflectance / NDVI")
            axes[0].legend(loc='upper right')
            axes[0].grid(alpha=0.3)
            axes[0].set_title(f"异常诊断: 反射率 & NDVI 曲线 - 日期: {d}")

            # 下图: 辐射量（RED & NIR）—— 【已优化：完美复刻你的截图配色与样式】
            axes[1].plot(sub['时间数值'], sub[E_up_red], marker='o', markersize=4, label=f'E_up Red {red_wv:.1f}nm', color='orange', linestyle='-')
            axes[1].plot(sub['时间数值'], sub[E_down_red], marker='o', markersize=4, label=f'E_down Red {red_wv:.1f}nm', color='red', linestyle='--')
            axes[1].plot(sub['时间数值'], sub[E_up_nir], marker='o', markersize=4, label=f'E_up NIR {nir_wv:.1f}nm', color='green', linestyle='-')
            axes[1].plot(sub['时间数值'], sub[E_down_nir], marker='o', markersize=4, label=f'E_down NIR {nir_wv:.1f}nm', color='blue', linestyle='--')
            axes[1].set_ylabel("E (W/m²/sr)")
            axes[1].legend(loc='upper right')
            axes[1].grid(alpha=0.3)
            axes[1].set_title(f"异常诊断: 辐射量曲线 (向上 & 向下) - 日期: {d}")
            axes[1].set_xlabel("一天中的时间（小时，小数制格式）")

            plt.tight_layout()

            # 将图像写入 PDF
            pdf.savefig(fig)

            # 同时保存单张图片到单独目录 (方便直接查看)
            day_dir = save_dir / "异常天数图表集"
            day_dir.mkdir(exist_ok=True, parents=True)
            plt.savefig(day_dir / f"Abnormal_{d}_诊断图.png", dpi=200)

            plt.close(fig)

    print(f"📄 PDF 报告及图片集已生成至：{save_dir}")
else:
    print("🎉 太棒了，未检测到任何异常数据！无需生成异常诊断报告。")