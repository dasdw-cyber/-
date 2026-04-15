import numpy as np
import pandas as pd
import matplotlib

# ==============================================================================
# 【关键修复】 解决 PyCharm/服务器环境下的绘图后端问题
# ==============================================================================
try:
    matplotlib.use('TkAgg')
except ImportError:
    pass

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import time
import os
from scipy.interpolate import interp1d

# 引入进度条
try:
    from tqdm import tqdm
except ImportError:
    print("提示: 未安装 tqdm 库，建议在终端运行 'pip install tqdm' 以显示进度条。")


    def tqdm(iterable, **kwargs):
        return iterable

try:
    from Py6S import *
except ImportError:
    print("错误: 未找到 Py6S 库。请确保已安装 Py6S wrapper 和 6S 可执行文件。")
    print("pip install Py6S")
    exit(1)


# ==========================================
# 1. 核心物理模型函数
# ==========================================

def gaussian_distribution(x, mu, fwhm):
    """计算高斯分布值 (SRF 权重)"""
    sigma = fwhm / 2.355
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def resample_reflectance(spectrum_wavelengths, spectrum_values, target_center, target_fwhm):
    """将全波段光谱重采样到目标传感器的波段"""
    spectrum_wavelengths = np.array(spectrum_wavelengths)
    spectrum_values = np.array(spectrum_values)

    center_nm = target_center * 1000
    fwhm_nm = target_fwhm * 1000

    min_w = center_nm - 3 * fwhm_nm
    max_w = center_nm + 3 * fwhm_nm

    data_min = spectrum_wavelengths.min()
    data_max = spectrum_wavelengths.max()

    if min_w < data_min or max_w > data_max:
        idx = (np.abs(spectrum_wavelengths - center_nm)).argmin()
        return spectrum_values[idx]

    grid_w = np.arange(min_w, max_w, 0.1)  # 高精度插值网格 0.1nm
    f_interp = interp1d(spectrum_wavelengths, spectrum_values, kind='linear', bounds_error=False,
                        fill_value='extrapolate')
    grid_refl = f_interp(grid_w)

    weights = gaussian_distribution(grid_w, center_nm, fwhm_nm)
    resampled_value = np.average(grid_refl, weights=weights)

    return resampled_value


def run_6s_simulation_with_srf(center_wavelength, fwhm, visibility, sza, ground_refl_value):
    """运行带有光谱响应函数 (SRF) 的 6S 模拟"""
    s = SixS()
    s.geometry = Geometry.User()
    s.geometry.solar_z = sza
    s.geometry.solar_a = 0
    s.geometry.view_z = 0
    s.geometry.view_a = 0

    # 【核心修复】必须使用 PredefinedType 实例化气溶胶对象，否则 6S 会静默降级为“无气溶胶”导致能见度参数失效
    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
    s.visibility = visibility

    # 动态输入该波段对应的裸土反射率 (强制转为标准 float，防止 numpy float 导致 Py6S 报错)
    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(float(ground_refl_value))

    # 对于 Visionpoint 这种极窄带(1nm)设备，直接启用单色波长模式
    if fwhm <= 0.0025:
        s.wavelength = Wavelength(center_wavelength)
    else:
        step = 0.0025
        range_width = fwhm * 3.0
        w_min = np.floor((center_wavelength - range_width) / step) * step
        wavelengths = np.arange(w_min, w_min + range_width * 2 + step, step)
        w_max = wavelengths[-1]
        filter_values = gaussian_distribution(wavelengths, center_wavelength, fwhm)
        s.wavelength = Wavelength(w_min, w_max, filter_values)

    s.run()
    return s.outputs.direct_solar_irradiance, s.outputs.diffuse_solar_irradiance


def run_6s_par_simulation(visibility, sza, ground_refl_value):
    """计算 PAR (400-700nm) 的模拟"""
    s = SixS()
    s.geometry = Geometry.User()
    s.geometry.solar_z = sza
    s.geometry.solar_a = 0
    s.geometry.view_z = 0
    s.geometry.view_a = 0

    # 【核心修复】同步修复 PAR 积分时的气溶胶实例化语法
    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
    s.visibility = visibility

    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(float(ground_refl_value))
    s.wavelength = Wavelength(0.400, 0.700)

    s.run()
    return s.outputs.direct_solar_irradiance, s.outputs.diffuse_solar_irradiance


def calculate_diffuse_fraction(direct, diffuse):
    total = direct + diffuse
    if total == 0: return 0
    return diffuse / total


def process_simulation_task(args):
    """执行单次模拟任务 (仅包含指定的 9 个波段)"""
    vis, sza, sensor_cfg, refl_dict, par_refl = args
    result = {'visibility': vis, 'sza': sza}

    try:
        # 1. 计算 PAR 的散射比例
        dir_par, dif_par = run_6s_par_simulation(vis, sza, par_refl)
        f_par = calculate_diffuse_fraction(dir_par, dif_par)
        result['f_par'] = f_par

        # 2. 遍历 Visionpoint 设备的 9 个特定波段
        for band_name, center_wl in sensor_cfg['centers'].items():
            val = refl_dict.get(band_name, 0.0)
            dir_b, dif_b = run_6s_simulation_with_srf(center_wl, sensor_cfg['fwhm'], vis, sza, val)
            result[f'visionpoint_{band_name}'] = calculate_diffuse_fraction(dir_b, dif_b)

        return result
    except Exception as e:
        # 打印底层报错以便排查
        print(f"\n[警告] visibility={vis}, sza={sza} 模拟失败: {e}")
        return None


# ==========================================
# 主程序
# ==========================================
if __name__ == '__main__':

    # ---------------------------------------------------------
    # 1. 读取真实土壤背景反射率 CSV
    # ---------------------------------------------------------
    csv_path = r"E:\叶绿素反演-李文娟老师论文\原始论文中的波段\白马土壤数据\24年小麦播种前裸土\PROSAIL_Trimmed_Soil_Background.csv"

    print(f"正在读取真实的土壤背景数据: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}。请检查路径。")
        exit(1)

    try:
        df_soil = pd.read_csv(csv_path, encoding='utf-8-sig')
        col_idx = 0 if df_soil.shape[1] == 1 else 1

        soil_values = pd.to_numeric(df_soil.iloc[:, col_idx], errors='coerce').dropna().values

        if np.max(soil_values) > 1.0:
            soil_values = soil_values / 100.0

        if df_soil.shape[1] > 1:
            soil_wavelengths = pd.to_numeric(df_soil.iloc[:, 0], errors='coerce').dropna().values
        else:
            soil_wavelengths = np.arange(440, 440 + len(soil_values))

        if len(soil_wavelengths) == 0:
            raise ValueError("未能从 CSV 中提取到有效的数值数据。")

        print(f"成功读取数据! 波长范围: {soil_wavelengths.min():.1f}nm - {soil_wavelengths.max():.1f}nm")
    except Exception as e:
        print(f"读取 CSV 失败: {e}")
        exit(1)

    # ---------------------------------------------------------
    # 2. 定制 Visionpoint 特定特征波段 (9个)
    # ---------------------------------------------------------
    target_bands_nm = [455, 545, 571, 615, 641, 662, 706, 728, 756]

    visionpoint_config = {
        'fwhm': 0.001,  # FWHM 设为 1nm
        'centers': {f"{int(wl)}nm": wl / 1000.0 for wl in target_bands_nm}
    }

    print(f"\n已配置 Visionpoint 特定反演特征波段 (共 {len(target_bands_nm)} 个)。")
    print("正在提取背景等效反射率...")

    BAND_REFLECTANCE = {}
    for band, center_um in visionpoint_config['centers'].items():
        refl = resample_reflectance(soil_wavelengths, soil_values, center_um, visionpoint_config['fwhm'])
        BAND_REFLECTANCE[band] = refl
        print(f"  [{band}] 土壤底层反射率: {refl:.4f}")

    par_indices = np.where((soil_wavelengths >= 400) & (soil_wavelengths <= 700))
    if len(par_indices[0]) > 0:
        PAR_REFLECTANCE = np.mean(soil_values[par_indices])
    else:
        PAR_REFLECTANCE = 0.0
    print(f"  [PAR (400-700nm)] 平均底层反射率: {PAR_REFLECTANCE:.4f}")
    print("=" * 60)

    # ---------------------------------------------------------
    # 3. 启动 6S 批量模拟 (切换为单进程安全模式)
    # ---------------------------------------------------------
    visibilities = np.arange(2.0, 42, 2)  # 能见度: 20 个值
    szas = np.arange(20, 70, 5)  # 太阳天顶角: 10 个值

    tasks = []
    for vis in visibilities:
        for sza in szas:
            tasks.append((vis, sza, visionpoint_config, BAND_REFLECTANCE, PAR_REFLECTANCE))

    print(f"\n🚀 准备开始 6S 模拟 (仅限指定的 9 个波段)，总任务数: {len(tasks)}")
    print("⚠️  为了彻底解决 6S Fortran 核心在 Windows 上的并发文件冲突问题，已切换为安全单进程模式。")
    print("⏳ 预计耗时约 3~5 分钟，请耐心等待...")

    start_time = time.time()
    results = []

    # 放弃不稳定的多进程，采用带进度条的稳健单进程遍历
    for task in tqdm(tasks, desc="6S 辐射传输模拟中", ncols=80):
        res = process_simulation_task(task)
        if res is not None:
            results.append(res)

    elapsed = time.time() - start_time
    print(f"✅ 模拟全部完成！成功获取 {len(results)} 个有效数据，总耗时: {elapsed:.2f} 秒")

    df = pd.DataFrame(results)

    if df.empty:
        print("\n❌ 严重错误: 所有 6S 模拟任务均失败！请检查上方输出的警告信息。")
        exit(1)

    df = df.sort_values(by='f_par').reset_index(drop=True)

    # ---------------------------------------------------------
    # 4. 拟合多项式系数 (打印 + 导出 CSV)
    # ---------------------------------------------------------
    print(f"\n=== Visionpoint 设备 9个特定波段 5次多项式拟合系数 ===")
    print(f"适用条件: f_par < 0.9 (论文原文限定)")
    print(f"{'Band':<8} | {'Polynomial Coefficients (5th degree: a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f)'}")
    print("-" * 90)

    models = {}
    coeff_data = []
    df_fit = df[df['f_par'] < 0.9]

    for band_name in visionpoint_config['centers'].keys():
        col_name = f"visionpoint_{band_name}"
        if col_name not in df_fit.columns: continue

        x = df_fit['f_par']
        y = df_fit[col_name]

        if len(x) > 10:
            coeffs = np.polyfit(x, y, 5)
            models[band_name] = np.poly1d(coeffs)

            coeff_data.append([band_name] + list(coeffs))
            coeffs_str = ", ".join([f"{c:8.4f}" for c in coeffs])
            print(f"{band_name:<8} | {coeffs_str}")

    output_dir = r'E:\叶绿素反演-李文娟老师论文\新采用的9波段\基于6s的公式系数'
    os.makedirs(output_dir, exist_ok=True)

    csv_out_path = os.path.join(output_dir, '9_bands_coefficients.csv')
    coeff_df = pd.DataFrame(coeff_data, columns=['Band', 'x^5', 'x^4', 'x^3', 'x^2', 'x^1', 'Intercept'])
    coeff_df.to_csv(csv_out_path, index=False)
    print(f"\n🎉 成功！这 9 个波段的拟合系数已同时保存至文件: {csv_out_path}")

    # ---------------------------------------------------------
    # 5. 可视化生成 (完整展示 9 个波段)
    # ---------------------------------------------------------
    print("\n正在绘制可视化图表...")
    try:
        plt.rcParams.update({'font.size': 12})
        fig, ax = plt.subplots(figsize=(8, 8))
        cmap = plt.get_cmap('jet_r')

        ax.plot([0, 1], [0, 1], 'k-', linewidth=1.5, label='1:1 Line', zorder=1)
        x_range = np.linspace(df_fit['f_par'].min(), 0.9, 100)

        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'd']
        marker_map = dict(zip(visionpoint_config['centers'].keys(), markers))

        for band_name in visionpoint_config['centers'].keys():
            if band_name in models:
                y_fit = models[band_name](x_range)
                ax.plot(x_range, y_fit, color='gray', linewidth=1.5, alpha=0.6, zorder=2)

        for band_name in visionpoint_config['centers'].keys():
            col_name = f"visionpoint_{band_name}"
            if col_name in df.columns:
                sc = ax.scatter(df['f_par'], df[col_name],
                                c=df['visibility'], cmap=cmap,
                                marker=marker_map[band_name],
                                s=45, alpha=0.8, edgecolors='none',
                                label=band_name, zorder=3)

        ax.set_xlabel(r'$f_{PAR}$ from 6S model')
        ax.set_ylabel(r'$f_{\lambda}$ from 6S model')
        ax.set_title('Simulated Diffuse Fraction (9 Specific Feature Bands)')
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle='--', alpha=0.3)

        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label('Visibility (km)')

        legend_elements = []
        for band_name, m in marker_map.items():
            line = mlines.Line2D([], [], color='white', marker=m,
                                 markerfacecolor='gray', markeredgecolor='black',
                                 markersize=8, label=band_name)
            legend_elements.append(line)
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True)

        plt.tight_layout()
        plt.show()
        print("图表已生成！")

    except Exception as e:
        print(f"图表显示警告: {e}")
        fig_out_path = os.path.join(output_dir, 'figure_6_visionpoint_9bands.png')
        plt.savefig(fig_out_path)
        print(f"已将图表保存为 {fig_out_path}")