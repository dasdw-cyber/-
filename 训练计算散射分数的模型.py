import pandas as pd
import numpy as np
import matplotlib

try:
    matplotlib.use('TkAgg')
except ImportError:
    pass
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os


def train_fpar_model(data_path=r"E:\叶绿素反演-李文娟老师论文\基于物理公式获取计算散射分数的模型的数据\step1_simulated_weather_data_spitters.csv", model_save_path=r'E:\叶绿素反演-李文娟老师论文\散射分数模型'):
    """
    训练 Step 1 的神经网络: 从 PAR_Total 估算 f_PAR
    """
    # 1. 准备工作
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据文件 {data_path}")
        print("请先运行 generate_step1_data.py 生成模拟数据。")
        return

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    print(f"正在加载数据: {data_path} ...")
    df = pd.read_csv(data_path)

    # 2. 定义输入特征 (X) 和 目标变量 (y)
    # 根据论文 Section 3.1: 输入为 Total PAR, Cos(Theta_S), d(t)
    feature_cols = ['Total_PAR', 'Cos_Theta_S', 'd_t']
    target_col = 'f_PAR'

    X = df[feature_cols].values
    y = df[target_col].values

    # 3. 数据集划分
    # 论文: 69659 (Train) / 14928 (Val) -> 大约 82% / 18%
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. 数据标准化 (Standardization)
    # 神经网络对输入数据的尺度非常敏感，必须标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 保存标准化器，后续预测时必须用同样的参数缩放输入
    joblib.dump(scaler, os.path.join(model_save_path, 'scaler_step1.pkl'))

    # 5. 训练循环 (Training Loop)
    # 论文提到训练了10个网络并选择RMSE最小的一个
    best_rmse = float('inf')
    best_model = None
    best_idx = -1

    print("\n开始训练 10 个候选模型 (结构: 4个隐藏神经元, Tanh激活)...")

    for i in range(10):
        # 定义模型: BP-ANN
        # hidden_layer_sizes=(4,): 1个隐藏层，4个神经元
        # activation='tanh': Tangent-Sigmoid 传输函数 (论文指定)
        # solver='adam': 优化器 (论文未指定，但Adam通常最稳健)
        # max_iter=500: 最大迭代次数
        model = MLPRegressor(hidden_layer_sizes=(4,),
                             activation='tanh',
                             solver='adam',
                             learning_rate_init=0.001,
                             max_iter=1000,
                             random_state=None)  # 让每次初始化不同

        # 训练
        model.fit(X_train_scaled, y_train)

        # 验证
        y_pred_val = model.predict(X_val_scaled)
        # 截断预测值到合理范围 [0, 1] (物理约束)
        y_pred_val = np.clip(y_pred_val, 0, 1)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        r2 = r2_score(y_val, y_pred_val)

        print(f"  Model {i + 1}: RMSE = {rmse:.4f}, R2 = {r2:.4f}")

        # 记录最佳模型
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_idx = i

    print(f"\n选出的最佳模型是: Model {best_idx + 1} (RMSE={best_rmse:.4f})")

    # 6. 保存最佳模型
    model_filename = os.path.join(model_save_path, 'bp_ann_df_model.pkl')
    joblib.dump(best_model, model_filename)
    print(f"模型已保存至: {model_filename}")

    # 7. 可视化评估 (类似论文 Fig. 5)
    visualize_results(best_model, scaler, X_val, y_val)


def visualize_results(model, scaler, X_val, y_val):
    """绘制验证集的 1:1 对比图"""
    print("\n正在生成评估图表...")

    # 预测
    X_val_scaled = scaler.transform(X_val)
    y_pred = model.predict(X_val_scaled)
    y_pred = np.clip(y_pred, 0, 1)

    # 计算最终指标
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    rrmse = (rmse / np.mean(y_val)) * 100

    # 绘图
    plt.figure(figsize=(7, 6))

    # 密度散点图 (使用Hexbin模拟论文效果)
    plt.hexbin(y_val, y_pred, gridsize=50, cmap='viridis', mincnt=1)
    cb = plt.colorbar(label='Count')

    # 1:1 线
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)

    # 添加统计信息
    stats_text = f"$R^2 = {r2:.2f}$\nRMSE = {rmse:.3f}\nRRMSE = {rrmse:.1f}%"
    plt.text(0.65, 0.1, stats_text, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    plt.xlabel('Measured PAR diffuse fraction')
    plt.ylabel('Estimated PAR diffuse fraction')
    plt.title('Step 1 Validation: BP-ANN-DF Performance')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()

    save_path =r'E:\叶绿素反演-李文娟老师论文\散射分数模型\step1_validation_plot.png'
    plt.savefig(save_path)
    print(f"评估图表已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    train_fpar_model()