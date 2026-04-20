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

# 🌟 1. 你的最终结果 CSV 路径 (你刚才手动修改过的文件)
RESULT_CSV_PATH = r"E:\叶绿素反演-李文娟老师论文\新9波段+LAI\反演_result\反演_引入实测LAI\Final_Inversion_Results_2026_Experiment.xlsx"

# 🌟 2. 实测 LAI 表格路径 (仅用于在主图上画星星和箭头做标记)
MEASURED_LAI_PATH = r"E:\叶绿素反演-李文娟老师论文\26年白马实测数据\2026白马LAI.xlsx"

# 🌟 3. 出图保存目录
OUTPUT_DIR = r"E:\叶绿素反演-李文娟老师论文\新9波段+LAI\反演_result\反演_引入实测LAI"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_YEAR = 2026

# 6个波段的名称定义 (用于画附属的核参数图)
BAND_NAMES = ['610nm', '680nm', '730nm', '760nm', '810nm', '860nm']
n_bands = len(BAND_NAMES)

# 如果你跑的是“串联反演法(Cascaded)”，这里的列名可能是 pred_cab_base 和 pred_cab_cascaded。
# 根据你 CSV 表头的实际名字修改下面这两个变量：
COL_CAB_BASE = 'pred_cab_27'  # 纯核参数反演的基准线
COL_CAB_HYBRID = 'pred_cab_hybrid'  # 加入LAI修正/串联约束后的红线


# ============================================================

def main():
    print(f"🎨 启动独立绘图程序，正在读取数据: {RESULT_CSV_PATH}")

    if not os.path.exists(RESULT_CSV_PATH):
        print("❌ 找不到 CSV 结果文件，请检查 RESULT_CSV_PATH 路径！")
        return

    # 1. 加载反演结果数据
    res_df = pd.read_excel(RESULT_CSV_PATH)
    res_df['date'] = pd.to_datetime(res_df['date'])
    res_df = res_df.sort_values('date').reset_index(drop=True)

    # 2. 加载实测 LAI (用于标记修正点)
    lai_map = {}
    if os.path.exists(MEASURED_LAI_PATH):
        lai_df = pd.read_excel(MEASURED_LAI_PATH)
        lai_df['日期_str'] = pd.to_datetime(lai_df['日期'].astype(str), format='%Y%m%d', errors='coerce').dt.strftime(
            '%Y-%m-%d')
        lai_map = dict(zip(lai_df['日期_str'].dropna(), lai_df['LAI'].dropna()))
    else:
        print("⚠️ 找不到 LAI 实测表格，将不绘制星星标记和箭头。")

    # =========================================================================
    # 图表一：主图 - 补偿效应对比图
    # =========================================================================
    print("📈 正在绘制图表 1: Cab 对比主图...")
    plt.figure(figsize=(10, 6))

    # 绘制基础线 (蓝色虚线)
    if COL_CAB_BASE in res_df.columns:
        plt.plot(res_df['date'], res_df[COL_CAB_BASE], 'b--s', linewidth=2, markersize=5, alpha=0.6,
                 label='Cab 预测值 (无 LAI 约束基准线)')
    else:
        print(f"❌ 找不到列 {COL_CAB_BASE}，请检查 CSV 表头！")

    # 绘制修正线 (红色实线)
    if COL_CAB_HYBRID in res_df.columns:
        plt.plot(res_df['date'], res_df[COL_CAB_HYBRID], 'r-o', linewidth=2, markersize=6,
                 label='Cab 预测值 (融合 LAI 强约束)')

    # 画星星和修正箭头
    for d_str, val in lai_map.items():
        point_data = res_df[res_df['date'] == pd.to_datetime(d_str)]
        if not point_data.empty:
            x_val = point_data['date'].values[0]
            y_val = point_data[COL_CAB_HYBRID].values[0]
            y_base = point_data[COL_CAB_BASE].values[0]

            # 只有当两个值不一样时（说明发生了修正），才画箭头
            if abs(y_val - y_base) > 0.1:
                plt.scatter(x_val, y_val, color='gold', marker='*', s=300, edgecolors='black', zorder=5)
                plt.annotate('', xy=(x_val, y_val), xytext=(x_val, y_base),
                             arrowprops=dict(arrowstyle="->", color='black', lw=1.5, ls=':'))
                plt.text(x_val, y_val - 3, f"{y_val:.1f}", ha='center', va='top', fontweight='bold', color='#d62728')

    plt.ylim(0, 70)
    plt.xlabel('日期')
    plt.ylabel('Cab ($\mu g/cm^2$)')
    plt.title(f'{TARGET_YEAR}年叶绿素含量反演时间序列对比图', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Cab_Comparison_Time_Series_{TARGET_YEAR}_Replot.png"), dpi=300)
    plt.close()

    # =========================================================================
    # 图表二：附图一 - 按核参数分类时间序列（点线图）
    # =========================================================================
    print("📈 正在绘制图表 2: 核参数分类图 (3x1)...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
    kernel_types = ['k0', 'k1', 'k2']
    titles = ['各向同性散射核参数 (k0)', '体散射核参数 (k1)', '几何光学核参数 (k2)']
    colors = plt.cm.tab10(np.linspace(0, 1, n_bands))

    for i, k_type in enumerate(kernel_types):
        ax = axes[i]
        for j, band in enumerate(BAND_NAMES):
            col_name = f'{band}_{k_type}'
            if col_name in res_df.columns:
                ax.plot(res_df['date'], res_df[col_name], marker='o', markersize=4, linestyle='-', linewidth=1.5,
                        color=colors[j], label=band)
        ax.set_title(titles[i], fontweight='bold')
        ax.set_ylabel('参数值')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    axes[-1].set_xlabel('日期')
    for ax in axes: ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Kernel_Parameters_By_Type_{TARGET_YEAR}_6Bands_Replot.png"), dpi=300,
                bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 图表三：附图二 - 按波段分类时间序列 (自动适配6波段为 2行3列)
    # =========================================================================
    print("📈 正在绘制图表 3: 波段分类图 (2x3)...")
    n_rows = (n_bands + 2) // 3
    fig_bands, axes_bands = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows), sharex=True)
    axes_bands = axes_bands.flatten()
    k_colors, k_labels = ['blue', 'green', 'red'], ['k0', 'k1', 'k2']

    for idx, band in enumerate(BAND_NAMES):
        ax = axes_bands[idx]
        for k_idx, k_type in enumerate(kernel_types):
            col_name = f'{band}_{k_type}'
            if col_name in res_df.columns:
                ax.plot(res_df['date'], res_df[col_name], marker='s', markersize=4, linestyle='-', linewidth=1.5,
                        color=k_colors[k_idx], label=k_labels[k_idx])
        ax.set_title(f'{band} 核参数动态变化', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best', fontsize='small')

    for i in range(len(axes_bands) - 3, len(axes_bands)):
        if i >= 0:
            axes_bands[i].set_xlabel('日期')
            axes_bands[i].tick_params(axis='x', rotation=45)

    for i in range(n_bands, len(axes_bands)):
        fig_bands.delaxes(axes_bands[i])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Kernel_Parameters_By_Band_{TARGET_YEAR}_6Bands_Replot.png"), dpi=300,
                bbox_inches='tight')
    plt.close()

    print(f"\n✅ 绘图大功告成！三张高清图表已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()