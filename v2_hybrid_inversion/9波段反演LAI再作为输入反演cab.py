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
import matplotlib.dates as mdates

# 忽略计算过程中的某些常见警告
warnings.filterwarnings('ignore')

# === 解决 matplotlib 中文显示方块乱码问题 ===
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ======================= 配置区域 =======================

REAL_DATA_PATH = r"E:\叶绿素反演-李文娟老师论文\白马光谱数据\2026年\NDVI_数据筛选结果\All_Data_总数据表.csv"

# 【模型路径】
MODEL_DIR_27 = r"E:\叶绿素反演-李文娟老师论文\新采用的9波段\反演Cab-ANN-model\核参数一起算\weiss参数\27核参数"
MODEL_DIR_28 = r"E:\叶绿素反演-李文娟老师论文\新9波段+LAI\反演cab-ANN-model\引入LAI测试"

COEFFS_CSV_FILE = r"E:\叶绿素反演-李文娟老师论文\新采用的9波段\基于6s的公式系数\9_bands_coefficients.csv"
OUTPUT_DIR = r"E:\叶绿素反演-李文娟老师论文\新9波段+LAI\反演_result-test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LATITUDE, LONGITUDE, TIME_ZONE = 31.62, 119.18, 8
VZA_FIXED, SENSOR_AZIMUTH = 45.0, 90.0
TARGET_YEAR = 2026

# 因为这是全自动化串联反演，无需依赖那3天的实测数据，因此放开全年的时间范围
START_DATE = "2026-03-01"
END_DATE = "2026-12-02"

print(f"正在读取 6S 动态多项式系数: {COEFFS_CSV_FILE}")
if not os.path.exists(COEFFS_CSV_FILE):
    raise FileNotFoundError("❌ 错误：找不到 9 波段多项式系数文件，请检查路径！")

coeff_df = pd.read_csv(COEFFS_CSV_FILE)
BAND_NAMES = coeff_df['Band'].tolist()
TARGET_BANDS = [int(name.replace('nm', '')) for name in BAND_NAMES]

F_LAMBDA_COEFFS = {}
for _, row in coeff_df.iterrows():
    F_LAMBDA_COEFFS[row['Band']] = [
        row['x^5'], row['x^4'], row['x^3'], row['x^2'], row['x^1'], row['Intercept']
    ]


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
    sza_r, vza_r, phi_r = np.radians(sza), np.radians(vza), np.radians(phi)
    cos_xi = np.cos(sza_r) * np.cos(vza_r) + np.sin(sza_r) * np.sin(vza_r) * np.cos(phi_r)
    phase = np.arccos(np.clip(cos_xi, -1.0, 1.0))
    term = (np.pi / 2.0 - phase) * np.cos(phase) + np.sin(phase)
    k_vol = (4.0 / (3.0 * np.pi)) * (term / (np.cos(sza_r) + np.cos(vza_r))) - (1.0 / 3.0)
    return k_vol


def roujean_k_geo(sza, vza, phi):
    sza_r, vza_r, phi_r = np.radians(sza), np.radians(vza), np.radians(phi)
    tan_s, tan_v = np.tan(sza_r), np.tan(vza_r)
    delta = np.sqrt(np.maximum(0, tan_s ** 2 + tan_v ** 2 - 2.0 * tan_s * tan_v * np.cos(phi_r)))
    term1 = (np.pi - phi_r) * np.cos(phi_r) + np.sin(phi_r)
    k_geo = (1.0 / (2.0 * np.pi)) * term1 * tan_s * tan_v - (1.0 / np.pi) * (tan_s + tan_v + delta)
    return k_geo


def integrate_diffuse_kernel_value():
    sza_range = np.linspace(0, 89, 90)
    phi_range = np.linspace(0, 359, 36)
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
    k_vol = roujean_k_vol(sza, vza, phi)
    k_geo = roujean_k_geo(sza, vza, phi)
    return (1 - f_lambda) * k_vol + f_lambda * K_VOL_DIFF, (1 - f_lambda) * k_geo + f_lambda * K_GEO_DIFF


# ======================= 主处理流程 =======================

def main():
    print("🚀 开始实测数据反演 (两步串联法：先反演LAI -> 构建27+LAI输入 -> 再反演Cab)...")

    # --- 1. 加载数据 ---
    df = pd.read_csv(REAL_DATA_PATH)
    df['datetime'] = pd.to_datetime(df['时间'])
    df = df.dropna(subset=['datetime']).copy()

    start_dt, end_dt = pd.to_datetime(START_DATE), pd.to_datetime(END_DATE) + pd.Timedelta(days=1) - pd.Timedelta(
        seconds=1)
    df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)].copy()

    # --- 2. 匹配波段与几何 ---
    refl_cols = [c for c in df.columns if "Reflectance-" in c]
    data_waves = [float(c.split('-')[1]) for c in refl_cols]
    selected_cols = {f"{t}nm": refl_cols[np.argmin(np.abs(np.array(data_waves) - t))] for t in TARGET_BANDS}

    for col in selected_cols.values():
        df = df[df[col] > 0]

    sza, saa, cos_sza, d_t, doy = calculate_solar_geometry_vectorized(df['datetime'], LATITUDE, LONGITUDE, TIME_ZONE)
    df['sza'], df['saa'], df['cos_sza'], df['d_t'] = sza, saa, cos_sza, d_t
    df['rel_phi'] = np.where(np.abs(saa - SENSOR_AZIMUTH) > 180, 360 - np.abs(saa - SENSOR_AZIMUTH),
                             np.abs(saa - SENSOR_AZIMUTH))

    # 动态 f_par 预测逻辑
    step1_model_path = os.path.join(MODEL_DIR_27, "bp_ann_df_model.pkl")
    step1_scaler_path = os.path.join(MODEL_DIR_27, "scaler_step1.pkl")
    if os.path.exists(step1_model_path) and os.path.exists(step1_scaler_path):
        model_df = joblib.load(step1_model_path)
        scaler_df = joblib.load(step1_scaler_path)
        total_par = df['E_up_Total'].values if 'E_up_Total' in df.columns else np.ones(len(df)) * 1000
        X_df = np.column_stack([total_par, df['cos_sza'].values, df['d_t'].values])
        X_df_scaled = scaler_df.transform(X_df)
        df['f_par'] = np.clip(model_df.predict(X_df_scaled), 0, 1)
    else:
        df['f_par'] = 0.2

    # --- 3. 滑动窗口核参数拟合 ---
    df['date_pd'] = pd.to_datetime(df['datetime'].dt.date)
    unique_dates = df['date_pd'].sort_values().unique()
    results = []
    n_bands = 9

    print(f"⏳ 正在进行滑动窗口 27 参数全局核特征拟合...")
    for current_date in unique_dates:
        window_df = df[(df['date_pd'] >= current_date - pd.Timedelta(days=1)) & (
                    df['date_pd'] <= current_date + pd.Timedelta(days=1))]
        window_df = window_df[(window_df['sza'] < 60) & (window_df['f_par'] > 0)]
        if len(window_df) < n_bands * 3: continue

        sza_arr, phi_arr, f_par_arr = window_df['sza'].values, window_df['rel_phi'].values, window_df['f_par'].values
        vza_arr = np.full_like(sza_arr, VZA_FIXED)
        day_coeffs = {'date': current_date, 'data_pts': len(window_df)}

        obs_abs_refls = np.array([window_df[selected_cols[bn]].values for bn in BAND_NAMES])
        obs_rel_refls = obs_abs_refls / (np.mean(obs_abs_refls, axis=0) + 1e-6)

        k_vol_dlc_all, k_geo_dlc_all = np.zeros_like(obs_abs_refls), np.zeros_like(obs_abs_refls)
        for i, bn in enumerate(BAND_NAMES):
            f_lam = np.clip(np.where(f_par_arr <= 0.9, np.polyval(F_LAMBDA_COEFFS[bn], f_par_arr), f_par_arr), 0, 1)
            k_vol_dlc_all[i], k_geo_dlc_all[i] = calculate_dlc_kernel(sza_arr, vza_arr, phi_arr, f_lam)

        def global_cost_func(params):
            p2d = params.reshape(n_bands, 3)
            mod_abs = np.array(
                [p2d[i, 0] + p2d[i, 1] * k_vol_dlc_all[i] + p2d[i, 2] * k_geo_dlc_all[i] for i in range(n_bands)])
            mod_rel = mod_abs / (np.mean(mod_abs, axis=0) + 1e-6)
            return np.sum((obs_rel_refls - mod_rel) ** 2) + 100.0 * np.sum((np.mean(mod_abs, axis=0) - 1.0) ** 2)

        x0 = [np.mean(obs_rel_refls[i, :]) for i in range(n_bands) for _ in range(3)]
        for i in range(n_bands): x0[i * 3 + 1] = 0.05; x0[i * 3 + 2] = 0.05

        res = minimize(global_cost_func, x0=x0, method='SLSQP', bounds=[(0, None), (-0.05, None), (-0.05, None)] * 9)
        if res.success:
            p_opt = res.x.reshape(n_bands, 3)
            for i, bn in enumerate(BAND_NAMES):
                day_coeffs[f'{bn}_k0'], day_coeffs[f'{bn}_k1'], day_coeffs[f'{bn}_k2'] = p_opt[i]
            results.append(day_coeffs)

    res_df = pd.DataFrame(results)

    # --- 4. 边界过滤 ---
    BOUNDS_JSON_PATH = os.path.join(MODEL_DIR_27, "kernel_bounds.json")
    if os.path.exists(BOUNDS_JSON_PATH):
        with open(BOUNDS_JSON_PATH, 'r') as f:
            kernel_bounds = json.load(f)
        valid_mask = np.ones(len(res_df), dtype=bool)
        for col, (k_min, k_max) in kernel_bounds.items():
            if col in res_df.columns: valid_mask &= (res_df[col] >= k_min) & (res_df[col] <= k_max)
        res_df = res_df[valid_mask].copy()

    # =========================================================================
    # 🌟🌟 5. 两步串联反演核心逻辑 (Two-step Cascaded Inversion) 🌟🌟
    # =========================================================================
    print("🧠 正在执行串联反演 (Step 1: 先反演LAI -> Step 2: 融合LAI再次反演Cab)...")

    feature_cols = [c for c in res_df.columns if '_k' in c]
    X_27_features = res_df[feature_cols].values

    # ---------------------------------------------------------
    # 【第一步】: 使用纯 27 参数模型反演 LAI 和 基础 Cab
    # ---------------------------------------------------------
    scaler_27 = joblib.load(os.path.join(MODEL_DIR_27, "scaler_inversion.pkl"))
    X_scaled_27 = scaler_27.transform(X_27_features)

    # 1.1 先把整条时间序列的 LAI 预测出来
    print("   -> [Step 1] 正在预测全时段 LAI...")
    lai_preds = []
    for i in range(10):
        m_path = os.path.join(MODEL_DIR_27, f"bp_ann_model_lai_{i}.pkl")
        lai_preds.append(np.maximum(joblib.load(m_path).predict(X_scaled_27), 0))
    res_df['pred_lai_27'] = np.median(np.column_stack(lai_preds), axis=1)

    # 1.2 预测基础 Cab（用于对比基准）
    cab_preds_27 = []
    for i in range(10):
        m_path = os.path.join(MODEL_DIR_27, f"bp_ann_model_cab_{i}.pkl")
        cab_preds_27.append(np.maximum(joblib.load(m_path).predict(X_scaled_27), 0))
    res_df['pred_cab_27'] = np.median(np.column_stack(cab_preds_27), axis=1)

    # ---------------------------------------------------------
    # 【第二步】: 构建 28 维联合特征 (27核参数 + 预测出来的LAI) 再次预测 Cab
    # ---------------------------------------------------------
    print("   -> [Step 2] 正在构建 28 维输入矩阵，执行全时段 Cab 强约束反演...")
    # np.column_stack 将矩阵横向拼接，实现 (N, 27) + (N, 1) = (N, 28)
    X_28_features = np.column_stack([X_27_features, res_df['pred_lai_27'].values])

    scaler_28 = joblib.load(os.path.join(MODEL_DIR_28, "scaler_inversion_with_lai.pkl"))
    X_scaled_28 = scaler_28.transform(X_28_features)

    cab_preds_cascaded = []
    for i in range(10):
        m_path = os.path.join(MODEL_DIR_28, f"bp_ann_model_cab_with_lai_{i}.pkl")
        cab_preds_cascaded.append(np.maximum(joblib.load(m_path).predict(X_scaled_28), 0))

    # 得到全时序列的“串联约束版 Cab”
    res_df['pred_cab_cascaded'] = np.median(np.column_stack(cab_preds_cascaded), axis=1)

    # =========================================================================
    # 6. 数据保存与合并对比预测折线图绘制
    # =========================================================================
    print("📊 正在生成最终图表...")
    res_df['date'] = pd.to_datetime(res_df['date'])
    res_df = res_df.sort_values('date').reset_index(drop=True)
    res_df.to_csv(os.path.join(OUTPUT_DIR, f"Final_Inversion_Results_{TARGET_YEAR}_Cascaded.csv"), index=False)

    # 🌟 主图：补偿效应全局串联对比图 (点线图)
    plt.figure(figsize=(10, 6))

    # 1. 纯27参基准线 (无约束)
    plt.plot(res_df['date'], res_df['pred_cab_27'], 'b--s', linewidth=2, markersize=5, alpha=0.6,
             label='第一步: 单纯使用27核反演 Cab (易受补偿效应影响)')

    # 2. 串联反演线 (带有预测 LAI 作为特征约束)
    plt.plot(res_df['date'], res_df['pred_cab_cascaded'], 'r-o', linewidth=2, markersize=6,
             label='第二步: 融合预测LAI联合反演 Cab (全局串联约束)')

    # 坐标系设置
    plt.ylim(0, 70)
    plt.xlabel('日期')
    plt.ylabel('Cab (μg/cm²)')
    plt.title(f'{TARGET_YEAR}年叶绿素串联反演算法 (Cascaded Inversion) 效果对比')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Cab_Cascaded_Time_Series_{TARGET_YEAR}.png"), dpi=300)
    plt.close()

    # =========================================================================
    # 7. 附图一：按核参数分类时间序列（点线图）
    # =========================================================================
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
    plt.savefig(os.path.join(OUTPUT_DIR, f"Kernel_Parameters_By_Type_{TARGET_YEAR}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 8. 附图二：按波段分类时间序列（点线图）
    # =========================================================================
    fig_bands, axes_bands = plt.subplots(3, 3, figsize=(18, 15), sharex=True)
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

    for i in range(6, 9):
        axes_bands[i].set_xlabel('日期')
        axes_bands[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Kernel_Parameters_By_Band_{TARGET_YEAR}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 串联反演全部完成！对比结果已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()