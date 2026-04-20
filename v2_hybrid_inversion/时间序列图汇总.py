import pandas as pd
import numpy as np
import os
import matplotlib

try:
    # 尝试使用兼容后端
    matplotlib.use('TkAgg')
except ImportError:
    pass
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 忽略常见警告
import warnings

warnings.filterwarnings('ignore')

# === 解决 matplotlib 中文显示方块乱码问题 ===
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ======================= 1. 配置区域 =======================

# 🌟 1. 9波段的最终结果路径 (请确认是 csv 还是 xlsx，代码已做自动兼容)
RESULT_9BANDS_PATH = r"E:\叶绿素反演-李文娟老师论文\新9波段+LAI\反演_result\反演_引入实测LAI\Final_Inversion_Results_2026_Experiment.xlsx"
COL_CAB_BASE_9 = 'pred_cab_27'  # 9波段的基准列名
COL_CAB_HYBRID_9 = 'pred_cab_hybrid'  # 9波段加LAI修正后的列名

# 🌟 2. 6波段的最终结果路径 (⚠️请修改为你电脑里 6 波段结果的实际路径)
RESULT_6BANDS_PATH = r"E:\叶绿素反演-李文娟老师论文\原始6波段+LAI\反演_result\引入实测LAI\Final_Inversion_Results_2026_Experiment.xlsx"
COL_CAB_BASE_6 = 'pred_cab_18'  # 6波段的基准列名 (如果是其他名字请修改)
COL_CAB_HYBRID_6 = 'pred_cab_hybrid'  # 6波段加LAI修正后的列名

# 🌟 3. 实测 LAI 表格路径 (用于画星星和箭头做标记)
MEASURED_LAI_PATH = r"E:\叶绿素反演-李文娟老师论文\26年白马实测数据\2026白马LAI.xlsx"

# 🌟 4. 出图保存目录
OUTPUT_DIR = r"E:\叶绿素反演-李文娟老师论文\引入LAI作为先验知识输入"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_YEAR = 2026


# ============================================================

def load_data(file_path):
    """自动判断后缀并加载数据"""
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return pd.DataFrame()

    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    return df


def main():
    print(f"🎨 启动双算法联合绘图程序...")

    # 1. 加载两个结果数据
    print("📥 正在读取 9波段 结果...")
    df9 = load_data(RESULT_9BANDS_PATH)

    print("📥 正在读取 6波段 结果...")
    df6 = load_data(RESULT_6BANDS_PATH)

    if df9.empty or df6.empty:
        print("❌ 数据加载失败，无法绘制对比图，请检查上面的路径配置！")
        return

    # 2. 加载实测 LAI
    lai_map = {}
    if os.path.exists(MEASURED_LAI_PATH):
        lai_df = pd.read_excel(MEASURED_LAI_PATH)
        lai_df['日期_str'] = pd.to_datetime(lai_df['日期'].astype(str), format='%Y%m%d', errors='coerce').dt.strftime(
            '%Y-%m-%d')
        lai_map = dict(zip(lai_df['日期_str'].dropna(), lai_df['LAI'].dropna()))
    else:
        print("⚠️ 找不到 LAI 实测表格，将不绘制星星标记和箭头。")

    # =========================================================================
    # 核心制图：双算法补偿效应对比图
    # =========================================================================
    print("📈 正在绘制 Cab 联合对比主图...")
    plt.figure(figsize=(12, 7))

    # ---------------- 绘制 6波段 曲线 ----------------
    if COL_CAB_BASE_6 in df6.columns:
        plt.plot(df6['date'], df6[COL_CAB_BASE_6], color='#2ca02c', marker='^', linestyle='--',
                 linewidth=1.5, markersize=4, alpha=0.4, label='6波段基准 (纯18核参数)')

    if COL_CAB_HYBRID_6 in df6.columns:
        plt.plot(df6['date'], df6[COL_CAB_HYBRID_6], color='#9467bd', marker='D', linestyle='-',
                 linewidth=2, markersize=6, alpha=0.9, label='6波段+LAI 约束模型')

    # ---------------- 绘制 9波段 曲线 ----------------
    if COL_CAB_BASE_9 in df9.columns:
        plt.plot(df9['date'], df9[COL_CAB_BASE_9], color='#1f77b4', marker='s', linestyle='--',
                 linewidth=1.5, markersize=4, alpha=0.4, label='9波段基准 (纯27核参数)')

    if COL_CAB_HYBRID_9 in df9.columns:
        plt.plot(df9['date'], df9[COL_CAB_HYBRID_9], color='#d62728', marker='o', linestyle='-',
                 linewidth=2.5, markersize=7, alpha=1.0, label='9波段+LAI 约束模型 (最精)')

    # ---------------- 画星星和修正箭头 ----------------
    for d_str, val in lai_map.items():
        target_date = pd.to_datetime(d_str)

        # 处理 6 波段的标记
        pt6 = df6[df6['date'] == target_date]
        if not pt6.empty and COL_CAB_HYBRID_6 in pt6.columns and COL_CAB_BASE_6 in pt6.columns:
            x_val = pt6['date'].values[0]
            y_val6 = pt6[COL_CAB_HYBRID_6].values[0]
            y_base6 = pt6[COL_CAB_BASE_6].values[0]
            if abs(y_val6 - y_base6) > 0.1:
                plt.scatter(x_val, y_val6, color='magenta', marker='*', s=200, edgecolors='black', zorder=5)
                plt.annotate('', xy=(x_val, y_val6), xytext=(x_val, y_base6),
                             arrowprops=dict(arrowstyle="->", color='#9467bd', lw=1.5, ls=':'))
                plt.text(x_val, y_val6 - 2.5, f"6波段:{y_val6:.1f}", ha='right', va='top', fontsize=9,
                         fontweight='bold', color='#9467bd')

        # 处理 9 波段的标记
        pt9 = df9[df9['date'] == target_date]
        if not pt9.empty and COL_CAB_HYBRID_9 in pt9.columns and COL_CAB_BASE_9 in pt9.columns:
            x_val = pt9['date'].values[0]
            y_val9 = pt9[COL_CAB_HYBRID_9].values[0]
            y_base9 = pt9[COL_CAB_BASE_9].values[0]
            if abs(y_val9 - y_base9) > 0.1:
                plt.scatter(x_val, y_val9, color='gold', marker='*', s=250, edgecolors='black', zorder=6)
                plt.annotate('', xy=(x_val, y_val9), xytext=(x_val, y_base9),
                             arrowprops=dict(arrowstyle="->", color='#d62728', lw=2, ls=':'))
                plt.text(x_val, y_val9 - 2.5, f"9波段:{y_val9:.1f}", ha='left', va='top', fontsize=9, fontweight='bold',
                         color='#d62728')

    # =========================================================================
    # 图表装帧
    # =========================================================================
    plt.ylim(10, 80)
    plt.xlabel('观测日期', fontsize=12, fontweight='bold')
    plt.ylabel('Cab 含量 ($\mu g/cm^2$)', fontsize=12, fontweight='bold')
    plt.title(f'{TARGET_YEAR}年 6波段 vs 9波段 联合反演对比及 LAI 约束效用评估', fontsize=15, fontweight='bold')

    # 图例优化，放到图外或选好位置避免遮挡
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='black')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"Cab_Comparison_6vs9_{TARGET_YEAR}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"\n✅ 双算法联合绘图大功告成！高清图表已保存至: {save_path}")


if __name__ == "__main__":
    main()