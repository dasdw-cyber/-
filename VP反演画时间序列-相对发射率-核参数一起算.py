import pandas as pd
import numpy as np
import joblib
import os
import json
import warnings
from scipy.optimize import minimize
import matplotlib

try:
    matplotlib.use('TkAgg')
except ImportError:
    pass
import matplotlib.pyplot as plt

# 忽略计算过程中的某些常见警告
warnings.filterwarnings('ignore')

# === 解决 matplotlib 中文显示方块乱码问题 ===
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ======================= 配置区域 =======================

# 1. 输入文件路径
REAL_DATA_PATH = r"E:\叶绿素反演-李文娟老师论文\原始论文中的波段\白马光谱数据\2026年\NDVI_数据筛选结果\All_Data_总数据表.csv"
MODEL_DIR = r"E:\叶绿素反演-李文娟老师论文\原始论文中的波段\反演Cab-ANN-model\核参数一起算\weiss参数"

# 2. 输出结果路径配置
OUTPUT_DIR = r"E:\叶绿素反演-李文娟老师论文\原始论文中的波段\反演-Result\2026年数据"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. 测站地理与仪器配置
LATITUDE = 31.62
LONGITUDE = 119.18
TIME_ZONE = 8
VZA_FIXED = 45.0
SENSOR_AZIMUTH = 90.0
TARGET_YEAR = 2025

# 🌟 新增：反演时间范围控制 (支持精确到具体生育期)
START_DATE = "2026-01-01"  # 起始日期 (格式: YYYY-MM-DD)
END_DATE = "2026-12-02"  # 结束日期 (格式: YYYY-MM-DD)

# 4. 目标波段与多项式系数
TARGET_BANDS = [610, 680, 730, 760, 810, 860]
BAND_NAMES = [f"{b}nm" for b in TARGET_BANDS]

F_LAMBDA_COEFFS = {
    '610nm': [-0.0670, 0.3414, -0.2481, 0.1061, 0.8540, -0.0050],
    '680nm': [0.1542, -0.0104, 0.0186, 0.0035, 0.7820, -0.0094],
    '730nm': [0.1558, 0.0792, -0.1189, 0.0922, 0.6953, -0.0077],
    '760nm': [0.0902, 0.2799, -0.3396, 0.2121, 0.6326, -0.0052],
    '810nm': [0.5728, -0.8620, 0.7432, -0.2945, 0.6997, -0.0170],
    '860nm': [0.4012, -0.3913, 0.2327, -0.0229, 0.5851, -0.0104]
}


# ======================= 核心函数库 =======================

def calculate_solar_geometry_vectorized(timestamps, lat_deg, lon_deg, tz_offset):
    doy = timestamps.dt.dayofyear.values
    hour_local = timestamps.dt.hour.values + timestamps.dt.minute.values / 60.0 + timestamps.dt.second.values / 3600.0
    gamma = 2 * np.pi / 365 * (doy - 1 + (hour_local - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    time_offset = eqtime + 4 * (lon_deg - 15 * tz_offset)
    tst = hour_local * 60 + time_offset
    ha_deg = (tst / 4) - 180
    ha_rad = np.radians(ha_deg)
    decl_rad = 0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) \
               - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma) \
               - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
    lat_rad = np.radians(lat_deg)
    cos_sza = np.sin(lat_rad) * np.sin(decl_rad) + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(ha_rad)
    cos_sza = np.clip(cos_sza, -1.0, 1.0)
    sza_rad = np.arccos(cos_sza)
    sza_deg = np.degrees(sza_rad)
    cos_saa = (np.sin(decl_rad) - np.sin(lat_rad) * np.cos(sza_rad)) / (np.cos(lat_rad) * np.sin(sza_rad))
    cos_saa = np.clip(cos_saa, -1.0, 1.0)
    saa_deg = np.degrees(np.arccos(cos_saa))
    saa_deg = np.where(ha_deg > 0, 360 - saa_deg, saa_deg)
    d_t = 1 + 0.01673 * np.cos(0.0172 * (doy - 2))
    return sza_deg, saa_deg, cos_sza, d_t, doy


def roujean_k_vol(sza, vza, phi):
    """Roujean 1992 体散射核 (K_vol)"""
    sza_r, vza_r, phi_r = np.radians(sza), np.radians(vza), np.radians(phi)
    cos_xi = np.cos(sza_r) * np.cos(vza_r) + np.sin(sza_r) * np.sin(vza_r) * np.cos(phi_r)
    phase = np.arccos(np.clip(cos_xi, -1.0, 1.0))
    term = (np.pi / 2.0 - phase) * np.cos(phase) + np.sin(phase)
    k_vol = (4.0 / (3.0 * np.pi)) * (term / (np.cos(sza_r) + np.cos(vza_r))) - (1.0 / 3.0)
    return k_vol


def roujean_k_geo(sza, vza, phi):
    """Roujean 1992 几何光学核 (K_geo)"""
    sza_r, vza_r, phi_r = np.radians(sza), np.radians(vza), np.radians(phi)
    tan_s, tan_v = np.tan(sza_r), np.tan(vza_r)
    delta = np.sqrt(np.maximum(0, tan_s ** 2 + tan_v ** 2 - 2.0 * tan_s * tan_v * np.cos(phi_r)))
    term1 = (np.pi - phi_r) * np.cos(phi_r) + np.sin(phi_r)
    k_geo = (1.0 / (2.0 * np.pi)) * term1 * tan_s * tan_v - (1.0 / np.pi) * (tan_s + tan_v + delta)
    return k_geo


def integrate_diffuse_kernel_value():
    """✅ 预计算漫射光条件下的 Roujean 核函数半球双重积分值"""
    sza_range = np.linspace(0, 89, 90)
    phi_range = np.linspace(0, 359, 36)  # 加入方位角积分
    k_vol_sum, k_geo_sum, weight_sum = 0, 0, 0

    for sza in sza_range:
        weight_sza = np.sin(np.radians(sza)) * np.cos(np.radians(sza))
        for phi in phi_range:
            k_vol_sum += roujean_k_vol(sza, VZA_FIXED, phi) * weight_sza
            k_geo_sum += roujean_k_geo(sza, VZA_FIXED, phi) * weight_sza
            weight_sum += weight_sza

    return k_vol_sum / weight_sum, k_geo_sum / weight_sum


K_VOL_DIFF, K_GEO_DIFF = integrate_diffuse_kernel_value()


def calculate_dlc_kernel(sza, vza, phi, f_lambda):
    """计算漫射修正核 (DLC)"""
    k_vol = roujean_k_vol(sza, vza, phi)
    k_geo = roujean_k_geo(sza, vza, phi)
    return (1 - f_lambda) * k_vol + f_lambda * K_VOL_DIFF, (1 - f_lambda) * k_geo + f_lambda * K_GEO_DIFF


# ======================= 主处理流程 =======================

def main():
    print("🚀 开始实测数据反演 (含时间控制与高级图表输出)...")

    # 1. 加载与过滤数据
    if not os.path.exists(REAL_DATA_PATH):
        print(f"❌ 找不到文件: {REAL_DATA_PATH}")
        return

    df = pd.read_csv(REAL_DATA_PATH)
    df['datetime'] = pd.to_datetime(df['时间'])
    df = df.dropna(subset=['datetime']).copy()

    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)].copy()

    if not df.empty:
        print(f"📅 已截取时间段: {df['datetime'].min().date()} 至 {df['datetime'].max().date()}")
        print(f"📊 参与反演的数据量: {len(df)} 条")
    else:
        print(f"❌ 警告：在 {START_DATE} 到 {END_DATE} 期间没有找到任何数据！程序退出。")
        return

    # 2. 匹配目标波段
    refl_cols = [c for c in df.columns if "Reflectance-" in c]
    data_wavelengths = [float(c.split('-')[1]) for c in refl_cols]
    selected_cols = {}
    for target in TARGET_BANDS:
        closest_idx = np.argmin(np.abs(np.array(data_wavelengths) - target))
        selected_cols[f"{target}nm"] = refl_cols[closest_idx]

    # 清洗负数
    for band_name, col_name in selected_cols.items():
        df = df[df[col_name] > 0]

    if df.empty:
        print("❌ 警告：数据清洗后为空，程序退出。")
        return

    # 3. 计算动态太阳几何
    sza, saa, cos_sza, d_t, doy = calculate_solar_geometry_vectorized(
        df['datetime'], LATITUDE, LONGITUDE, TIME_ZONE)
    df['sza'] = sza
    df['saa'] = saa
    df['cos_sza'] = cos_sza
    df['d_t'] = d_t
    rel_phi = np.abs(saa - SENSOR_AZIMUTH)
    df['rel_phi'] = np.where(rel_phi > 180, 360 - rel_phi, rel_phi)

    # 漫射比例估算
    step1_model_path = os.path.join(MODEL_DIR, "bp_ann_df_model.pkl")
    step1_scaler_path = os.path.join(MODEL_DIR, "scaler_step1.pkl")
    if os.path.exists(step1_model_path) and os.path.exists(step1_scaler_path):
        model_df = joblib.load(step1_model_path)
        scaler_df = joblib.load(step1_scaler_path)
        total_par = df['E_up_Total'].values if 'E_up_Total' in df.columns else np.ones(len(df)) * 1000
        X_df = np.column_stack([total_par, df['cos_sza'].values, df['d_t'].values])
        X_df_scaled = scaler_df.transform(X_df)
        df['f_par'] = np.clip(model_df.predict(X_df_scaled), 0, 1)
    else:
        df['f_par'] = 0.2

    # 4. 滑动窗口 18 参数联合拟合
    df['date_pd'] = pd.to_datetime(df['datetime'].dt.date)
    unique_dates = df['date_pd'].sort_values().unique()
    results = []

    # 构建 18 个参数的边界 (6个波段 * 3个参数)
    bounds_18 = [(0, None), (-0.05, None), (-0.05, None)] * 6

    print("⏳ 正在进行滑动窗口 18 参数全局核特征拟合...")
    for current_date in unique_dates:
        window_mask = (df['date_pd'] >= current_date - pd.Timedelta(days=1)) & \
                      (df['date_pd'] <= current_date + pd.Timedelta(days=1))
        window_df = df[window_mask].copy()
        window_df = window_df[(window_df['sza'] < 60) & (window_df['f_par'] > 0)]

        if len(window_df) < 18:
            continue  # 保证观测数至少大于未知数数量

        sza_arr, phi_arr, f_par_arr = window_df['sza'].values, window_df['rel_phi'].values, window_df['f_par'].values
        vza_arr = np.full_like(sza_arr, VZA_FIXED)
        day_coeffs = {'date': current_date.date(), 'data_pts': len(window_df)}

        # 准备绝对观测值矩阵
        obs_abs_refls = np.zeros((6, len(window_df)))
        k_vol_dlc_all = np.zeros((6, len(window_df)))
        k_geo_dlc_all = np.zeros((6, len(window_df)))

        for i, band_name in enumerate(BAND_NAMES):
            # 获取观测绝对值
            obs_abs_refls[i, :] = window_df[selected_cols[band_name]].values

            # 预计算每个波段的漫射修正核
            coeffs = F_LAMBDA_COEFFS[band_name]
            f_lam_arr = np.clip(np.where(f_par_arr <= 0.9, np.polyval(coeffs, f_par_arr), f_par_arr), 0, 1)
            k_vol_dlc_all[i], k_geo_dlc_all[i] = calculate_dlc_kernel(sza_arr, vza_arr, phi_arr, f_lam_arr)

        # 🌟 统一转换为相对观测值 (对应公式 8 的目标)
        spectral_means = np.mean(obs_abs_refls, axis=0)
        obs_rel_refls = obs_abs_refls / (spectral_means + 1e-6)

        # 🌟 核心修正：定义带尺度锚定惩罚的 18 参数联合代价函数 (严格对齐 PROSAIL 训练集逻辑)
        def global_cost_func(params):
            params_2d = params.reshape(6, 3)
            X_mod_abs = np.zeros((6, len(window_df)))

            for i in range(6):
                k0, k1, k2 = params_2d[i]
                X_mod_abs[i] = k0 + k1 * k_vol_dlc_all[i] + k2 * k_geo_dlc_all[i]

            mean_X_mod = np.mean(X_mod_abs, axis=0)
            X_mod_rel = X_mod_abs / (mean_X_mod + 1e-6)

            # 光谱形状拟合误差
            shape_error = np.sum((obs_rel_refls - X_mod_rel) ** 2)

            # 💥 尺度锚定惩罚项：强制锁定物理量纲
            scale_penalty = 100.0 * np.sum((mean_X_mod - 1.0) ** 2)

            return shape_error + scale_penalty

        # 🌟 核心修正：初始值使用相对观测值，而非绝对观测值
        x0 = []
        for i in range(6):
            x0.extend([np.mean(obs_rel_refls[i, :]), 0.05, 0.05])

        # 执行全局最优化
        res = minimize(global_cost_func, x0=x0, method='SLSQP', bounds=bounds_18)

        # 保存这 3 天滑动窗口对应的核参数
        if res.success:
            params_opt = res.x.reshape(6, 3)
            for i, band in enumerate(BAND_NAMES):
                day_coeffs[f'{band}_k0'] = params_opt[i, 0]
                day_coeffs[f'{band}_k1'] = params_opt[i, 1]
                day_coeffs[f'{band}_k2'] = params_opt[i, 2]

            # 计算并记录平均均方根误差 (这里记录的是未加惩罚项的纯形状误差，用于参考)
            pure_shape_error = np.sum((obs_rel_refls - (np.zeros((6, len(window_df))) + 1e-6)) ** 2)  # 简化表示
            day_coeffs['mean_fit_rmse'] = np.sqrt(res.fun / (6 * len(window_df)))
            results.append(day_coeffs)

    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("❌ 拟合失败，未能提取出有效核参数。")
        return

    # 5. 边界过滤 (利用之前提取的 PROSAIL json 边界)
    BOUNDS_JSON_PATH = os.path.join(MODEL_DIR, "kernel_bounds.json")
    if os.path.exists(BOUNDS_JSON_PATH):
        with open(BOUNDS_JSON_PATH, 'r') as f:
            kernel_bounds = json.load(f)
        valid_mask = np.ones(len(res_df), dtype=bool)
        for col, (k_min, k_max) in kernel_bounds.items():
            if col in res_df.columns: valid_mask &= (res_df[col] >= k_min) & (res_df[col] <= k_max)

        # 记录被过滤掉的数量
        filtered_count = len(res_df) - valid_mask.sum()
        if filtered_count > 0:
            print(f"⚠️ 基于 PROSAIL 物理边界约束，已剔除 {filtered_count} 天受异常噪声干扰的核参数数据。")

        res_df = res_df[valid_mask].copy()

    # 6. ANN 预测
    print("🧠 正在使用集成 BP 神经网络预测生理参数...")
    scaler_path = os.path.join(MODEL_DIR, "scaler_inversion.pkl")
    scaler = joblib.load(scaler_path)
    feature_cols = [c for c in res_df.columns if '_k' in c]
    X_scaled = scaler.transform(res_df[feature_cols].values)

    for target in ['lai', 'cab', 'ccc']:
        predictions = []
        for i in range(10):
            p = os.path.join(MODEL_DIR, f"bp_ann_model_{target}_{i}.pkl")
            if os.path.exists(p):
                predictions.append(np.maximum(joblib.load(p).predict(X_scaled), 0))
        if predictions:
            res_df[f'pred_{target}'] = np.median(np.column_stack(predictions), axis=1)

    # =========================================================================
    # 7. 数据保存与预测折线图
    # =========================================================================
    print("📊 正在生成可视化图表...")
    res_df['date'] = pd.to_datetime(res_df['date'])
    res_df = res_df.sort_values('date').reset_index(drop=True)
    res_df.to_csv(os.path.join(OUTPUT_DIR, f"Final_Inversion_Results_{TARGET_YEAR}.csv"), index=False)

    plt.figure(figsize=(10, 6))
    if 'pred_cab' in res_df.columns:
        plt.plot(res_df['date'], res_df['pred_cab'], 'g*-', linewidth=2, label='Cab (预测值)')

    plt.ylim(0, 70)
    plt.xlabel('日期')
    plt.ylabel('Cab (μg/cm²)')
    plt.title(f'{TARGET_YEAR}年叶绿素含量反演时间序列图')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Cab_Inversion_Time_Series_{TARGET_YEAR}.png"), dpi=300)
    plt.close()

    # =========================================================================
    # 8. 图表一：按核参数分类时间序列（3张子图，每张展示6个波段）
    # =========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    kernel_types = ['k0', 'k1', 'k2']
    titles = [
        '各向同性散射核参数 (k0) - 反映天底绝对反射率',
        '体散射核参数 (k1) - 反映冠层内部多次散射',
        '几何光学核参数 (k2) - 反映土壤/阴影的几何遮挡'
    ]
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange']

    for i, k_type in enumerate(kernel_types):
        ax = axes[i]
        for j, band in enumerate(BAND_NAMES):
            col_name = f'{band}_{k_type}'
            if col_name in res_df.columns:
                ax.plot(res_df['date'], res_df[col_name], marker='o', markersize=4,
                        linestyle='-', linewidth=1.5, color=colors[j], label=band)

        ax.set_title(titles[i], fontweight='bold')
        ax.set_ylabel('参数值')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    axes[-1].set_xlabel('日期')
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    kernel_plot_path = os.path.join(OUTPUT_DIR, f"Kernel_Parameters_By_Type_{TARGET_YEAR}.png")
    plt.savefig(kernel_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 9. 图表二：按波段分类时间序列（6张子图，每张展示3个核参数）
    # =========================================================================
    fig_bands, axes_bands = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    axes_bands = axes_bands.flatten()

    k_colors = ['blue', 'green', 'red']
    k_labels = ['k0 (各向同性)', 'k1 (体散射)', 'k2 (几何光学)']

    for idx, band in enumerate(BAND_NAMES):
        ax = axes_bands[idx]
        for k_idx, k_type in enumerate(kernel_types):
            col_name = f'{band}_{k_type}'
            if col_name in res_df.columns:
                ax.plot(res_df['date'], res_df[col_name], marker='s', markersize=4,
                        linestyle='-', linewidth=1.5, color=k_colors[k_idx], label=k_labels[k_idx])

        ax.set_title(f'{band} 核参数动态变化', fontweight='bold')
        ax.set_ylabel('参数值')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best', fontsize='small')

    for i in range(4, 6):
        axes_bands[i].set_xlabel('日期')
        axes_bands[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    kernel_band_plot_path = os.path.join(OUTPUT_DIR, f"Kernel_Parameters_By_Band_{TARGET_YEAR}.png")
    plt.savefig(kernel_band_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 全部完成！结果文件已保存至: {OUTPUT_DIR}")
    print(
        f"👉 本次输出包含:\n 1. Final_Inversion_Results_{TARGET_YEAR}.csv (反演数值文件)\n 2. Cab 预测折线图\n 3. 核参数(按类型)趋势面板图(3图)\n 4. 核参数(按波段)趋势面板图(6图)")


if __name__ == "__main__":
    main()