import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

# 忽略一些不必要的警告
warnings.filterwarnings('ignore')

# ==========================================
# 设置中文字体，防止图表中的中文显示为方块
# 根据您的操作系统，可能需要调整字体名称（Windows常用'SimHei'，Mac常用'Arial Unicode MS'）
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_chlorophyll_time_series():
    # 1. 读取数据文件 (加入新的27核参数反演结果)
    file1_path = r"E:\叶绿素反演-李文娟老师论文\原始论文中的波段\利用LICI指数算LCC\反演验证结果_3SV\VP_清洗与LCC反演明细_2025年.csv"
    file2_path = r"E:\叶绿素反演-李文娟老师论文\原始论文中的波段\反演-Result\反演结果-withsoil-select\with-noise\roujean\weiss参数设置\特征约束\18核参数一起算\windows3\Final_Inversion_Results_2025.csv"
    file3_path = r"E:\叶绿素反演\25暑假白马数据\spad-cab\水稻小麦数据汇总2.xlsx"
    # 新增：9波段 27核参数的反演结果
    file4_path = r"E:\叶绿素反演-李文娟老师论文\新采用的9波段\反演-Result\weiss参数设置\27核参数一起算\windows3\Final_Inversion_Results_2025_27Params.csv"

    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        df4 = pd.read_csv(file4_path)  # 读取新加入的27参数数据
        # 读取实测 Excel 数据 (需要环境中安装有 openpyxl 库)
        df3 = pd.read_excel(file3_path)
    except FileNotFoundError as e:
        print(f"文件读取失败，请检查文件路径: {e}")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        print("💡 提示: 读取 xlsx 文件需要 openpyxl 库。如果没有安装，请在终端运行: pip install openpyxl")
        return

    # 2. 解析时间并进行聚合

    # 解析 VP 明细数据并求日均值
    df1['Parsed_Time'] = pd.to_datetime(df1['时间'])
    df1['Date'] = df1['Parsed_Time'].dt.normalize()
    agg_dict = {'LCC_Estimated_3SV': 'mean'}
    df_daily1 = df1.groupby('Date').agg(agg_dict).reset_index().sort_values('Date')

    # 解析 File2 最终结果数据 (18核参数)
    df2['date'] = pd.to_datetime(df2['date'])
    df2 = df2.sort_values('date')

    # 解析 File4 最终结果数据 (27核参数)
    df4['date'] = pd.to_datetime(df4['date'])
    df4 = df4.sort_values('date')

    # 解析 File3 实测数据的时间
    if pd.api.types.is_datetime64_any_dtype(df3['日期']):
        df3['Date'] = df3['日期'].apply(lambda dt: dt.replace(year=2025))
    else:
        # 清理可能被读取为浮点数或整数的日期简写
        date_strs = df3['日期'].astype(str).str.replace(r'\.0$', '', regex=True)
        # 补齐4位并加上2025年转为datetime
        df3['Date'] = pd.to_datetime('2025' + date_strs.str.zfill(4), format='%Y%m%d', errors='coerce')

    # 3. 开始绘图：同轴对比
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=150)

    # ----------------- 【主图】同轴对比模型估算值与实测值 -----------------
    ax1.set_xlabel('采样日期', fontsize=12)
    ax1.set_ylabel('冠层叶绿素含量 ($\mu g/cm^2$)', fontsize=12)

    # 获取Y轴范围 (如果有异常值，建议放开限制或自动获取)
    ax1.set_ylim(0, 80)

    # [曲线 1] 绘制 File1 (VP明细 LICI) - Crimson 虚线风格
    ax1.plot(df_daily1['Date'], df_daily1['LCC_Estimated_3SV'],
             linestyle='--', color='crimson', linewidth=2,
             label='经验指数 (LICI)', alpha=0.8)

    # [曲线 2] 绘制 File2 (18参数 反演) - 宝蓝色 实线
    ax1.plot(df2['date'], df2['pred_cab'],
             linestyle='-', color='royalblue', linewidth=2,
             label='PROSAIL反演 (18核参数)', alpha=0.8)

    # [曲线 3] 绘制 File4 (27参数 反演) - 深橙色 点划线
    ax1.plot(df4['date'], df4['pred_cab'],
             linestyle='-.', color='darkorange', linewidth=2,
             label='PROSAIL反演 (9波段-27核参数)', alpha=0.9)

    # [散点 4] 绘制 File3 (实测 Ground Truth) - 保留极大、醒目的标记
    if 'cab' in df3.columns:
        ax1.scatter(df3['Date'], df3['cab'],
                    color='red',  # 醒目的红色填充
                    edgecolor='black',  # 黑色描边增加对比度
                    linewidths=1.5,  # 边框加粗
                    marker='*',  # 五角星形状
                    s=500,  # 极大的尺寸
                    zorder=10,  # 保证悬浮在所有图层最顶端
                    label='实测真值 (Ground Truth)')

    ax1.set_title('2025年白马试验基地：多重模型反演冠层叶绿素与实测值对比', fontsize=15, pad=15)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 调整图例位置，避免遮挡数据
    ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=10)

    # X轴格式化
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # 优化布局并展示/保存
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    plt.savefig('Time_Series_Comparison_27Params.png')
    print("✅ 时间序列对比图(已加入27核参数模型)已保存至: Time_Series_Comparison_27Params.png")
    plt.show()


if __name__ == "__main__":
    plot_chlorophyll_time_series()