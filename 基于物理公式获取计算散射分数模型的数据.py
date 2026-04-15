import pandas as pd
import numpy as np
import datetime
import pytz
from pysolar.solar import get_altitude, get_azimuth

# ================= 配置区域 =================
# 地点: 南京白马农业高新技术产业示范区
LATITUDE = 31.62
LONGITUDE = 119.19
LOCAL_TIMEZONE = 'Asia/Shanghai'

# 传感器几何
SENSOR_ZENITH = 45.0  # 观测天顶角 固定 45度
SENSOR_AZIMUTH = 180.0  # 假设传感器朝南

# 输出文件路径 (严格使用你的本地路径)
OUTPUT_FILE = r"E:\叶绿素反演-李文娟老师论文\基于物理公式获取计算散射分数的模型的数据\step1_simulated_weather_data_spitters.csv"


# ================= 核心函数 =================

def generate_nanjing_schedule():
    """
    """
    timestamps = []

    # 设定播种到收获的时间跨度
    start_date = datetime.date(2024, 11, 12)
    end_date = datetime.date(2025, 12, 31)
    delta = datetime.timedelta(days=1)

    current_date = start_date
    while current_date <= end_date:
        # 每天从早 8 点到下午 17 点
        for hour in range(8, 18):
            dt = datetime.datetime(current_date.year, current_date.month, current_date.day, hour, 0, 0)
            timestamps.append(dt)
        current_date += delta

    return pd.DataFrame({'timestamp': timestamps})


def calculate_solar_geometry(row):
    """
    计算精准的天文几何角度 (PROSAIL 格式)
    """
    local_time = row['timestamp']
    if local_time.tzinfo is None:
        local_tz = pytz.timezone(LOCAL_TIMEZONE)
        local_time = local_tz.localize(local_time)

    utc_time = local_time.astimezone(pytz.utc)

    # 1. 计算太阳位置 (基于真实地理位置和时间)
    sun_alt = get_altitude(LATITUDE, LONGITUDE, utc_time)
    sun_az = get_azimuth(LATITUDE, LONGITUDE, utc_time)

    # 2. 转换为 PROSAIL 角度
    tts = 90.0 - sun_alt
    tto = SENSOR_ZENITH

    # PSI (相对方位角)
    az_diff = abs(sun_az - SENSOR_AZIMUTH)
    if az_diff > 180:
        az_diff = 360 - az_diff
    psi = abs(az_diff - 180)

    # 3. 计算辅助参数
    doy = local_time.timetuple().tm_yday
    d_t = 1 + 0.01673 * np.cos(0.0172 * (doy - 2))

    return pd.Series([tts, tto, psi, sun_az, doy, d_t])


def simulate_par_and_diffusion_physical(df):
    """
    【核心融合】基于天气的 Kt 生成 + Spitters (1986) 物理模型
    """
    np.random.seed(42)
    SC_PAR = 2500  # 太阳常数中PAR的近似值 (umol/m2/s)

    # 1. 每天随机分配一种基调天气状况
    dates = df['timestamp'].dt.date.unique()
    weather_map = {}
    for d in dates:
        weather_map[d] = np.random.choice(['sunny', 'cloudy', 'overcast'], p=[0.4, 0.4, 0.2])

    total_pars = []
    f_pars = []
    kts = []

    for idx, row in df.iterrows():
        c_tts = np.cos(np.deg2rad(row['tts']))

        # 剔除夜间或极低角度
        if c_tts <= 0.05:
            total_pars.append(0.0)
            f_pars.append(1.0)
            kts.append(0.0)
            continue

        # 2. 计算大气层顶理想 PAR (TOA PAR)
        toa_par = SC_PAR * row['d_t'] * c_tts

        # 3. 🌟 修改点 2：根据基准天气生成大气透射率 Kt，并引入小时级的突发云层遮挡
        base_weather = weather_map[row['timestamp'].date()]
        if base_weather == 'sunny':
            kt = np.random.uniform(0.65, 0.85)
            # 15% 概率出现一团云遮挡太阳，导致直射光骤降
            if np.random.rand() < 0.15:
                kt = np.random.uniform(0.3, 0.5)

        elif base_weather == 'cloudy':
            kt = np.random.uniform(0.35, 0.65)
            # 多云天气有 15% 概率云层突然变厚，或者偶尔放晴
            if np.random.rand() < 0.15:
                kt = np.random.uniform(0.1, 0.35) if np.random.rand() < 0.5 else np.random.uniform(0.65, 0.75)

        else:  # overcast
            kt = np.random.uniform(0.1, 0.35)

        # 4. 引入 Spitters 模型推导 f_par
        if kt < 0.07:
            f_par = 1.0
        elif kt < 0.35:
            f_par = 1.0 - 2.3 * (kt - 0.07) ** 2
        elif kt < 0.75:
            f_par = 1.33 - 1.46 * kt
        else:
            f_par = 0.23

        # 添加设备测量噪声，防止神经网络过拟合
        noise = np.random.normal(0, 0.05)
        f_par = np.clip(f_par + noise, 0.1, 1.0)

        # 5. 计算到达地面的实际 Total PAR
        total_par = toa_par * kt

        total_pars.append(total_par)
        f_pars.append(f_par)
        kts.append(kt)

    df['Kt'] = kts
    df['Total_PAR'] = total_pars
    df['f_PAR'] = f_pars
    # 计算神经网络需要的特征
    df['Cos_Theta_S'] = np.cos(np.deg2rad(df['tts']))
    return df


# ================= 主程序 =================

def run_step1():
    print("--- Step 1: 生成南京白马基地天气与几何数据 (增强物理融合版) ---")

    print("1. 生成时间表 (冬小麦全生育期, 每天 8-17点)...")
    df = generate_nanjing_schedule()
    print(f"   共生成观测点: {len(df)} 个")

    print("2. 计算精准几何角度...")
    geom_cols = df.apply(calculate_solar_geometry, axis=1)
    geom_cols.columns = ['tts', 'tto', 'psi', 'sun_azimuth', 'doy', 'd_t']
    df = pd.concat([df, geom_cols], axis=1)

    print("3. 过滤无效数据 (🌟 严格对齐论文：剔除天顶角 > 60 的数据)...")
    df = df[df['tts'] < 60].reset_index(drop=True)

    print("4. 使用 Spitters 模型推导并注入天气波动扰动...")
    df = simulate_par_and_diffusion_physical(df)

    print("5. 保存结果...")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"--- 完成! 数据已保存至: {OUTPUT_FILE} ---")

    print("\n--- 融合后的数据预览 ---")
    print(df[['timestamp', 'tts', 'Kt', 'Total_PAR', 'f_PAR']].head())


if __name__ == "__main__":
    try:
        run_step1()
    except ImportError:
        print("缺少必要的库，请运行: pip install pysolar pytz pandas numpy")