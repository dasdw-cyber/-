import pandas as pd
import numpy as np
import os
import glob
import warnings

# ================= 1. 路径配置 =================
input_folder = r"E:\叶绿素反演-李文娟老师论文\原始论文中的波段\白马光谱数据\2026年\原始数据"
calib_path = r"E:\LESS-project\VisionPoint使用文档\辐射定标参数-更正new2-南农.xls"

# 🌟【已修复】修正了之前复制重叠的冗长路径
output_dir_step1 = r"E:\叶绿素反演-李文娟老师论文\原始论文中的波段\白马光谱数据\2026年\辐射定标数据"
os.makedirs(output_dir_step1, exist_ok=True)

# ================= 2. 加载定标参数 =================
print("正在加载辐射定标参数...")
calib = pd.read_excel(calib_path)
calib = calib[(calib['wave(nm)'] > 423) & (calib['wave(nm)'] <= 921)].copy()

# 提取波段数值为浮点数，方便后续自定义格式化小数位
wavelengths_num = calib['wave(nm)'].values
k3 = calib['k3'].values
k4 = calib['k4'].values

# ================= 3. 批量定标处理 =================
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
print(f"检测到 {len(csv_files)} 个原始文件，开始辐射定标...")

for file_path in csv_files:
    file_name = os.path.basename(file_path)
    try:
        data = pd.read_csv(file_path)

        # 基础校验
        if not all(col in data.columns for col in ['对地曝光时间', '对天曝光时间', '时间']):
            continue

        # 强制拼接成 6 位小数的列名，匹配原始数据
        down_cols = [f"down-{wl:.6f}" for wl in wavelengths_num]
        up_cols = [f"up-{wl:.6f}" for wl in wavelengths_num]

        if any(c not in data.columns for c in down_cols):
            print(f"⚠️ 跳过 {file_name}: 表格中未找到匹配的列，例如 {down_cols[0]}")
            continue

        down_gray = data[down_cols]
        up_gray = data[up_cols]
        exp_down = data['对地曝光时间'].replace(0, np.nan)
        exp_up = data['对天曝光时间'].replace(0, np.nan)

        k3_df = pd.DataFrame(np.tile(k3, (len(down_gray), 1)), columns=down_gray.columns, index=down_gray.index)
        k4_df = pd.DataFrame(np.tile(k4, (len(up_gray), 1)), columns=up_gray.columns, index=up_gray.index)

        # 计算辐射量
        E_down = (down_gray * k3_df).div(exp_down, axis=0)
        E_up = (up_gray * k4_df).div(exp_up, axis=0)

        # 加入 np.pi 计算真实的半球-圆锥反射率
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reflectance_values = (np.pi * E_down.values) / E_up.values

        # 格式化反射率列名
        ref_cols = [f"Reflectance-{wl:.1f}" for wl in wavelengths_num]
        reflectance_df = pd.DataFrame(reflectance_values, columns=ref_cols, index=data.index)
        reflectance_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 🌟【核心修复】将 E_down 和 E_up 也转换为 DataFrame，并规范命名为保留 1 位小数的波段名
        e_down_cols = [f"E_down-{wl:.1f}" for wl in wavelengths_num]
        e_up_cols = [f"E_up-{wl:.1f}" for wl in wavelengths_num]

        E_down_df = pd.DataFrame(E_down.values, columns=e_down_cols, index=data.index)
        E_up_df = pd.DataFrame(E_up.values, columns=e_up_cols, index=data.index)

        # 拼接原始特征、辐射量(E_down/E_up) 与 反射率，剔除冗余的灰度列
        cols_to_drop = down_cols + up_cols
        file_result = pd.concat([data.drop(columns=cols_to_drop), E_down_df, E_up_df, reflectance_df], axis=1)

        # 保存定标结果
        save_path = os.path.join(output_dir_step1, f"Calibrated_{file_name}")
        file_result.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"✅ 定标完成并保存: {file_name}")

    except Exception as e:
        print(f"❌ 处理 {file_name} 时出错: {e}")

print("\n第一阶段：辐射定标全部完成！(已加入 π 值修正及辐射量输出)")