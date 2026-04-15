import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# ================= 配置路径 =================
input_path = r"E:\叶绿素反演-李文娟老师论文\白马光谱数据\NDVI_数据筛选结果\正确辐射定标\Normal_Data_仅std筛选.csv"
output_folder = r"E:\叶绿素反演-李文娟老师论文\白马土壤数据\25年水稻种植前泡田"
os.makedirs(output_folder, exist_ok=True)

# 目标日期 (25年裸土期)
target_dates = ["2025-06-08", "2025-06-10", "2025-06-12"]

try:
    print("正在加载并筛选数据...")
    df = pd.read_csv(input_path, low_memory=False)

    # 1. 时间解析与双重筛选 (日期 + 小时)
    df['Parsed_Time'] = pd.to_datetime(df['时间'], errors='coerce')
    df = df.dropna(subset=['Parsed_Time'])

    # 筛选指定日期
    df = df[df['Parsed_Time'].dt.strftime('%Y-%m-%d').isin(target_dates)]
    # 筛选 10:00 - 14:00 (含14:00整)
    df = df[(df['Parsed_Time'].dt.hour >= 10) & (df['Parsed_Time'].dt.hour <= 14)]

    if df.empty:
        print("❌ 筛选后无可用数据，请检查输入文件的时间列或时间段。")
    else:
        # ================= 统计实际使用的日期与数据量 =================
        df['Date_Only'] = df['Parsed_Time'].dt.strftime('%Y-%m-%d')
        actual_dates_counts = df['Date_Only'].value_counts().sort_index()
        actual_days_num = len(actual_dates_counts)
        total_spectra = len(df)

        print(f"\n✅ 目标日期中，实际包含有效数据(10:00-14:00)的有 {actual_days_num} 天：")
        for date_str, count in actual_dates_counts.items():
            print(f"   - {date_str}: 共 {count} 条光谱数据")
        print("--------------------------------------------------\n")

        # ================= 2. 提取并截断核心数据矩阵 (430-921nm) =================
        refl_cols = [c for c in df.columns if c.startswith('Reflectance-')]
        wavelengths = np.array([float(c.replace('Reflectance-', '')) for c in refl_cols])

        # 按波段排序
        sort_idx = np.argsort(wavelengths)
        wavelengths = wavelengths[sort_idx]
        refl_matrix = df[[refl_cols[i] for i in sort_idx]].values  # N行 x 波段数

        # 🔥 新增：波段截断掩码 (保留 430 到 921 之间的波段)
        valid_bands = (wavelengths >= 440) & (wavelengths <= 921)
        wavelengths = wavelengths[valid_bands]
        refl_matrix = refl_matrix[:, valid_bands]

        print(f"✂️ 波段已截断：保留 {wavelengths.min():.1f} nm 至 {wavelengths.max():.1f} nm 区间的数据。")

        # ================= 3. 自动剔除离群光谱 (Outlier Removal) =================
        # 计算每条光谱与中位数的平均绝对误差 (MAE)，剔除偏离最大的前 5%
        median_spectrum = np.median(refl_matrix, axis=0)
        mae_dist = np.mean(np.abs(refl_matrix - median_spectrum), axis=1)

        threshold = np.percentile(mae_dist, 95)
        valid_indices = mae_dist <= threshold
        clean_matrix = refl_matrix[valid_indices]

        dropped_count = len(refl_matrix) - len(clean_matrix)
        print(f"🧹 自动清洗：移除了 {dropped_count} 条离群异常光谱，保留 {len(clean_matrix)} 条健康光谱。")

        # ================= 4. 计算均值与 SG 平滑滤波 =================
        clean_mean_spectrum = np.mean(clean_matrix, axis=0)

        # 窗口设为 21，多项式阶数 3，有效去除残留毛刺
        smoothed_spectrum = savgol_filter(clean_mean_spectrum, window_length=21, polyorder=3)
        smoothed_spectrum = np.clip(smoothed_spectrum, 0, 1)  # 防止出现负值

        # ================= 5. 1nm 整数重采样 (适配 PROSAIL) =================
        min_w = int(np.ceil(wavelengths.min()))
        max_w = int(np.floor(wavelengths.max()))
        prosail_wavelengths = np.arange(min_w, max_w + 1)

        interp_func = interp1d(wavelengths, smoothed_spectrum, kind='cubic')
        prosail_reflectance = interp_func(prosail_wavelengths)

        # ================= 6. 输出 PROSAIL 格式 CSV =================
        prosail_df = pd.DataFrame({
            'Wavelength': prosail_wavelengths,
            'Reflectance': prosail_reflectance
        })

        # 输出极简两列格式，无索引，名字已修正
        save_csv = os.path.join(output_folder, "PROSAIL_Trimmed_Soil_Background.csv")
        prosail_df.to_csv(save_csv, index=False, encoding='utf-8-sig')
        print(f"✅ PROSAIL 专用背景已保存: {save_csv}")

        # ================= 7. 绘图诊断可视化 =================
        plt.figure(figsize=(12, 7))

        # 画出清洗后的单条数据作背景
        plt.plot(wavelengths, clean_matrix.T, color='cyan', alpha=0.15, linewidth=1)
        plt.plot([], [], color='cyan', alpha=0.5, label=f'Cleaned Individual (n={len(clean_matrix)})')

        # 原始均值 (虚线)
        plt.plot(wavelengths, np.mean(refl_matrix, axis=0), color='black', linestyle='--', linewidth=2,
                 label='Raw Mean')

        # PROSAIL 输入均值 (实线)
        plt.plot(prosail_wavelengths, prosail_reflectance, color='blue', linewidth=3,
                 label='PROSAIL Input (440-921nm)')

        plt.title(f"Trimmed Soil Background Diagnostic ({actual_days_num} Days, {total_spectra} Samples)", fontsize=15)
        plt.xlabel("Wavelength (nm)", fontsize=12)
        plt.ylabel("Reflectance", fontsize=12)

        # 强制设置X轴范围，让截断效果在图上更直观
        plt.xlim(420, 930)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper left', fontsize=11)
        plt.tight_layout()

        # 保存图片，名字已修正
        save_fig = os.path.join(output_folder, "Trimmed_Spectrum_Plot.png")
        plt.savefig(save_fig, dpi=300)
        print(f"📊 截断后背景诊断图已生成: {save_fig}")
        plt.show()

except Exception as e:
    print(f"💥 处理失败: {e}")