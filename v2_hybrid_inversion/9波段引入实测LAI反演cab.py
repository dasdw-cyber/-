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

REAL_DATA_PATH = r"E:\叶绿素反演-李文娟老师论文\白马光谱数据\2026年\NDVI_数据筛选结果\All_Data_总数据表.csv"
MEASURED_LAI_PATH = r"E:\叶绿素反演-李文娟老师论文\26年白马实测数据\2026白马LAI.xlsx"

MODEL_DIR_27 = r"E:\叶绿素反演-李文娟老师论文\新采用的9波段\反演Cab-ANN-model\核参数一起算\weiss参数\27核参数"
MODEL_DIR_28 = r"E:\叶绿素反演-李文娟老师论文\新9波段+LAI\反演cab-ANN-model\引入LAI测试"

COEFFS_CSV_FILE = r"E:\叶绿素反演-李文娟老师论文\新采用的9波段\基于6s的公式系数\9_bands_coefficients.csv"
OUTPUT_DIR = r"E:\叶绿素反演-李文娟老师论文\新9波段+LAI\反演_result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LATITUDE = 31.62
LONGITUDE = 119.18
TIME_ZONE = 8
VZA_FIXED = 45.0
SENSOR_AZIMUTH = 90.0
TARGET_YEAR = 2026

START_DATE = "2026-03-20"
END_DATE = "2026-04-15"

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
    print("🚀 开始实测数据反演 (保证与原版算法完全一致的 27核提取，附带LAI实验比对)...")

    # 1. 加载与过滤数据
    df = pd.read_csv(REAL_DATA_PATH)
    df['datetime'] = pd.to_datetime(df['时间'])
    df = df.dropna(subset=['datetime']).copy()

    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)].copy()

    # 读取实测 LAI (严格匹配日期)
    print(f"📖 读取实测 LAI 表格: {MEASURED_LAI_PATH}")
    lai_df = pd.read_excel(MEASURED_LAI_PATH)
    lai_df['日期_str'] = pd.to_datetime(lai_df['日期'].astype(str), format='%Y%m%d', errors='coerce').dt.strftime(
        '%Y-%m-%d')
    lai_map = dict(zip(lai_df['日期_str'].dropna(), lai_df['LAI'].dropna()))

    # 2. 匹配目标波段
    refl_cols = [c for c in df.columns if "Reflectance-" in c]
    data_wavelengths = [float(c.split('-')[1]) for c in refl_cols]
    selected_cols = {}
    for target in TARGET_BANDS:
        closest_idx = np.argmin(np.abs(np.array(data_wavelengths) - target))
        selected_cols[f"{target}nm"] = refl_cols[closest_idx]

    for band_name, col_name in selected_cols.items():
        df = df[df[col_name] > 0]

    # 3. 🌞 完全还原：计算动态太阳几何与 f_par
    sza, saa, cos_sza, d_t, doy = calculate_solar_geometry_vectorized(df['datetime'], LATITUDE, LONGITUDE, TIME_ZONE)
    df['sza'], df['saa'], df['cos_sza'], df['d_t'] = sza, saa, cos_sza, d_t
    rel_phi = np.abs(saa - SENSOR_AZIMUTH)
    df['rel_phi'] = np.where(rel_phi > 180, 360 - rel_phi, rel_phi)

    # 🌟 绝对保留：原版计算 f_par 的逻辑，绝不乱删！
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

    # 4. 📈 完全还原：滑动窗口 27 参数联合拟合
    df['date_pd'] = pd.to_datetime(df['datetime'].dt.date)
    unique_dates = df['date_pd'].sort_values().unique()
    results = []
    bounds_27 = [(0, None), (-0.05, None), (-0.05, None)] * 9
    n_bands = 9

    print(f"⏳ 正在进行滑动窗口 {n_bands * 3} 参数全局核特征拟合...")
    for current_date in unique_dates:
        window_mask = (df['date_pd'] >= current_date - pd.Timedelta(days=1)) & \
                      (df['date_pd'] <= current_date + pd.Timedelta(days=1))
        window_df = df[window_mask].copy()
        window_df = window_df[(window_df['sza'] < 60) & (window_df['f_par'] > 0)]

        if len(window_df) < n_bands * 3:
            continue

        sza_arr, phi_arr, f_par_arr = window_df['sza'].values, window_df['rel_phi'].values, window_df['f_par'].values
        vza_arr = np.full_like(sza_arr, VZA_FIXED)
        day_coeffs = {'date': current_date, 'data_pts': len(window_df)}  # 保留原来的 date 对象

        obs_abs_refls = np.zeros((n_bands, len(window_df)))
        k_vol_dlc_all = np.zeros((n_bands, len(window_df)))
        k_geo_dlc_all = np.zeros((n_bands, len(window_df)))

        for i, band_name in enumerate(BAND_NAMES):
            obs_abs_refls[i, :] = window_df[selected_cols[band_name]].values
            coeffs = F_LAMBDA_COEFFS[band_name]
            f_lam_arr = np.clip(np.where(f_par_arr <= 0.9, np.polyval(coeffs, f_par_arr), f_par_arr), 0, 1)
            k_vol_dlc_all[i], k_geo_dlc_all[i] = calculate_dlc_kernel(sza_arr, vza_arr, phi_arr, f_lam_arr)

        spectral_means = np.mean(obs_abs_refls, axis=0)
        obs_rel_refls = obs_abs_refls / (spectral_means + 1e-6)

        def global_cost_func(params):
            params_2d = params.reshape(n_bands, 3)
            X_mod_abs = np.zeros((n_bands, len(window_df)))
            for i in range(n_bands):
                k0, k1, k2 = params_2d[i]
                X_mod_abs[i] = k0 + k1 * k_vol_dlc_all[i] + k2 * k_geo_dlc_all[i]
            mean_X_mod = np.mean(X_mod_abs, axis=0)
            X_mod_rel = X_mod_abs / (mean_X_mod + 1e-6)
            shape_error = np.sum((obs_rel_refls - X_mod_rel) ** 2)
            scale_penalty = 100.0 * np.sum((mean_X_mod - 1.0) ** 2)
            return shape_error + scale_penalty

        # 🌟 绝对保留：原版计算 x0 的代码
        x0 = []
        for i in range(n_bands):
            x0.extend([np.mean(obs_rel_refls[i, :]), 0.05, 0.05])

        res = minimize(global_cost_func, x0=x0, method='SLSQP', bounds=bounds_27)
        if res.success:
            params_opt = res.x.reshape(n_bands, 3)
            for i, band in enumerate(BAND_NAMES):
                day_coeffs[f'{band}_k0'] = params_opt[i, 0]
                day_coeffs[f'{band}_k1'] = params_opt[i, 1]
                day_coeffs[f'{band}_k2'] = params_opt[i, 2]
            results.append(day_coeffs)

    res_df = pd.DataFrame(results)

    # 5. 边界过滤
    BOUNDS_JSON_PATH = os.path.join(MODEL_DIR_27, "kernel_bounds.json")
    if os.path.exists(BOUNDS_JSON_PATH):
        with open(BOUNDS_JSON_PATH, 'r') as f:
            kernel_bounds = json.load(f)
        valid_mask = np.ones(len(res_df), dtype=bool)
        for col, (k_min, k_max) in kernel_bounds.items():
            if col in res_df.columns: valid_mask &= (res_df[col] >= k_min) & (res_df[col] <= k_max)
        res_df = res_df[valid_mask].copy()

    # =========================================================================
    # 6. ANN 预测 (A: 原版纯27计算 | B: 实测控制替换)
    # =========================================================================
    print("🧠 正在使用 ANN 进行反演...")
    scaler_path_27 = os.path.join(MODEL_DIR_27, "scaler_inversion.pkl")
    scaler_27 = joblib.load(scaler_path_27)
    feature_cols = [c for c in res_df.columns if '_k' in c]
    X_scaled_27 = scaler_27.transform(res_df[feature_cols].values)

    # A. 完全沿用原版的 27 参预测
    for target in ['lai', 'cab', 'ccc']:
        predictions = []
        for i in range(10):
            p = os.path.join(MODEL_DIR_27, f"bp_ann_model_{target}_{i}.pkl")
            if os.path.exists(p):
                predictions.append(np.maximum(joblib.load(p).predict(X_scaled_27), 0))
        if predictions:
            res_df[f'pred_{target}_27'] = np.median(np.column_stack(predictions), axis=1)

    # B. 双模型替换逻辑：仅增加对 pred_cab_hybrid 的更新，绝不触碰上面的原始计算
    res_df['pred_cab_hybrid'] = res_df['pred_cab_27'].copy()

    scaler_path_28 = os.path.join(MODEL_DIR_28, "scaler_inversion_with_lai.pkl")
    scaler_28 = joblib.load(scaler_path_28)
    models_28 = [joblib.load(os.path.join(MODEL_DIR_28, f"bp_ann_model_cab_with_lai_{i}.pkl")) for i in range(10)]

    for idx, row in res_df.iterrows():
        current_date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
        if current_date_str in lai_map:
            measured_lai = float(lai_map[current_date_str])
            print(f"   => 💥 命中实验日期: {current_date_str}, 引入实测 LAI={measured_lai} 进行局部约束")

            x_28_features = np.append(row[feature_cols].values.astype(float), measured_lai).reshape(1, -1)
            x_28_scaled = scaler_28.transform(x_28_features)

            cab_preds_28 = [np.maximum(m.predict(x_28_scaled), 0)[0] for m in models_28]
            res_df.loc[idx, 'pred_cab_hybrid'] = np.median(cab_preds_28)

    # =========================================================================
    # 7. 数据保存与绘图
    # =========================================================================
    print("📊 正在生成可视化图表...")
    res_df['date'] = pd.to_datetime(res_df['date'])
    res_df = res_df.sort_values('date').reset_index(drop=True)
    res_df.to_csv(os.path.join(OUTPUT_DIR, f"Final_Inversion_Results_{TARGET_YEAR}_Experiment.csv"), index=False)

    # 主图：补偿效应红蓝线对比图
    plt.figure(figsize=(10, 6))

    plt.plot(res_df['date'], res_df['pred_cab_27'], 'b--s', linewidth=2, markersize=5, alpha=0.6,
             label='Cab 预测值 (纯27核参数，含补偿误差)')
    plt.plot(res_df['date'], res_df['pred_cab_hybrid'], 'r-o', linewidth=2, markersize=6,
             label='Cab 预测值 (在实测日引入LAI强约束)')

    for d_str, val in lai_map.items():
        point_data = res_df[res_df['date'] == pd.to_datetime(d_str)]
        if not point_data.empty:
            x_val = point_data['date'].values[0]
            y_val = point_data['pred_cab_hybrid'].values[0]
            y_base = point_data['pred_cab_27'].values[0]

            plt.scatter(x_val, y_val, color='gold', marker='*', s=300, edgecolors='black', zorder=5)
            plt.annotate('', xy=(x_val, y_val), xytext=(x_val, y_base),
                         arrowprops=dict(arrowstyle="->", color='black', lw=1.5, ls=':'))
            plt.text(x_val, y_val - 3, f"{y_val:.1f}", ha='center', va='top', fontweight='bold', color='#d62728')

    plt.ylim(0, 70)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('Cab ($\mu g/cm^2$)', fontsize=12)
    plt.title(f'{TARGET_YEAR}年叶绿素含量反演时间序列对比图', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Cab_Comparison_Time_Series_{TARGET_YEAR}.png"), dpi=300)
    plt.close()

    print(f"\n✅ 绝对同步完成！对比结果已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()