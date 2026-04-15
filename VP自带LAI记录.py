import pandas as pd
import os


def extract_daily_lai():
    # 1. 配置文件路径与列名
    input_path = r"E:\叶绿素反演-李文娟老师论文\原始论文中的波段\白马光谱数据\2026年\NDVI_数据筛选结果\All_Data_总数据表.csv"
    target_col = '叶面积指数 m2·m-2'
    time_col = '时间'  # 根据您之前的数据结构，假设时间列名为 '时间'

    # 自动在同目录下生成输出文件
    output_path =r"E:\叶绿素反演-李文娟老师论文\VP自带数据\LAI\2026年\LAI.csv"

    print(f"📂 正在读取数据文件: {input_path}")

    try:
        # 2. 读取数据 (使用适合包含中文路径和内容的编码读取)
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print("❌ 错误：找不到指定的文件，请检查路径是否准确。")
        return

    # 检查目标列是否存在
    if target_col not in df.columns or time_col not in df.columns:
        print(f"❌ 错误：数据表中缺少 '{target_col}' 或 '{time_col}' 列，请检查表头。")
        return

    # 3. 时间序列转换
    # 将时间列转换为 datetime 格式，并剔除时间格式错误的空行
    df['datetime'] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=['datetime'])

    # 提取纯日期用于后续分组
    df['Date'] = df['datetime'].dt.date

    # 4. 精准筛选 10:00 - 14:00 的数据
    # 将 datetime 设置为索引后，使用 between_time 可以极速过滤出该时间段的数据
    df_indexed = df.set_index('datetime')
    df_filtered = df_indexed.between_time('10:00', '14:00').reset_index()

    # 剔除目标列中可能存在的空值，避免影响均值计算
    df_filtered = df_filtered.dropna(subset=[target_col])

    if df_filtered.empty:
        print("⚠️ 警告：在 10:00 - 14:00 时间段内没有找到任何有效的 LAI 数据。")
        return

    # 5. 按日期分组，计算均值
    print("⏳ 正在计算每日 10-14 点均值...")
    daily_lai = df_filtered.groupby('Date')[target_col].mean().reset_index()

    # 重命名列，让输出结果更清晰
    daily_lai.rename(columns={target_col: 'LAI_10至14点均值'}, inplace=True)

    # 对日期进行排序，确保时间序列连贯
    daily_lai = daily_lai.sort_values('Date').reset_index(drop=True)

    # 6. 导出结果
    # 使用 utf-8-sig 编码保存，确保在 Excel 中打开时表头中文不会乱码
    daily_lai.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"✅ 处理完成！共提取了 {len(daily_lai)} 天的数据。")
    print(f"📁 结果已保存至: {output_path}")


if __name__ == "__main__":
    extract_daily_lai()