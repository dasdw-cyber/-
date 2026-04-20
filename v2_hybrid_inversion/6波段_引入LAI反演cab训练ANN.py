import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib

try:
    # 尝试使用 PyCharm 兼容后端，或让其自动选择
    matplotlib.use('TkAgg')
except ImportError:
    pass
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ======================= 配置区域 =======================
# 🌟 PROSAIL 模拟数据集路径 (请确保这个数据集来源于6波段模拟，且包含18个核参数和lai真值)
DATA_PATH = r"E:\叶绿素反演-李文娟老师论文\原始论文中的6波段\Prosail-data\核参数一起计算\weiss参数\prosail_training_database_hybrid.csv"

# 🌟 模型保存目录 (指向我们刚才设定的 19输入带LAI约束 的专门文件夹)
MODEL_SAVE_DIR = r"E:\叶绿素反演-李文娟老师论文\原始6波段+LAI\反演cab-ANN-model\引入LAI测试"

# 🌟 目标变量列表 (因为 LAI 已经是已知输入特征了，所以这里只专心反演 cab)
TARGETS = ['cab']

# 每个变量训练的集成模型数量 (论文要求 10 个)
N_ENSEMBLE = 10


# =======================================================

def train_inversion_models():
    # 1. 准备工作
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误: 找不到训练数据文件: {DATA_PATH}")
        print("请先运行 PROSAIL 数据生成脚本。")
        return

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    print(f"📥 正在加载数据: {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    # 检查数据完整性，删除由拟合失败导致的空值行
    initial_len = len(df)
    df = df.dropna()
    if len(df) < initial_len:
        print(f"⚠️ 已移除 {initial_len - len(df)} 行含有空值的数据。")

    print(f"📊 有效训练样本数: {len(df)}")

    # 2. 提取特征 (X)
    # 先筛选 18 个核系数
    feature_cols = [c for c in df.columns if '_k' in c]

    # 💥💥 核心修改：强制把 LAI 加入输入特征 💥💥
    feature_cols.append('lai')

    if len(feature_cols) != 19:
        print(f"⚠️ 警告: 识别到的特征列数量为 {len(feature_cols)}，预期为 19 (18核参数 + 1个LAI)。")
        print(f"列名: {feature_cols}")

    X = df[feature_cols].values

    # 提取物理边界并保存为 JSON (此时会自动包含 LAI 的边界)
    print("🔍 正在提取 19 维特征 (核参数+LAI) 的物理边界...")
    bounds_dict = {}
    for col in feature_cols:
        bounds_dict[col] = [float(df[col].min()), float(df[col].max())]

    bounds_path = os.path.join(MODEL_SAVE_DIR, 'kernel_and_lai_bounds.json')
    with open(bounds_path, 'w') as f:
        json.dump(bounds_dict, f, indent=4)
    print(f"✅ 物理边界已轻量化保存至: {bounds_path}")

    # 3. 数据标准化 (Standardization)
    print("📏 正在进行 19 维数据标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 🌟 命名修改：保存为 scaler_inversion_with_lai.pkl
    scaler_path = os.path.join(MODEL_SAVE_DIR, 'scaler_inversion_with_lai.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"💾 标准化器已保存: {scaler_path}\n")

    # 4. 循环训练目标变量 (此处只有 cab)
    for target in TARGETS:
        if target not in df.columns:
            print(f"⚠️ 跳过 {target}: 列名不存在")
            continue

        print(f"{'=' * 15} 🚀 开始训练 {target.upper()} 的强约束集成模型 (N={N_ENSEMBLE}) {'=' * 15}")
        y = df[target].values

        # 划分训练集和验证集 (70% / 30% 根据你上一版的设定)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        ensemble_predictions = []

        # 10 个网络循环训练与持久化
        for i in range(N_ENSEMBLE):
            print(f"  ⏳ 正在训练 {target.upper()} (带LAI) 模型 {i + 1}/{N_ENSEMBLE} ...", end="")

            model = MLPRegressor(hidden_layer_sizes=(5,),
                                 activation='tanh',
                                 solver='lbfgs',
                                 max_iter=3000,
                                 early_stopping=False,
                                 random_state=i)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            y_pred = np.maximum(y_pred, 0)  # 物理约束截断

            ensemble_predictions.append(y_pred)

            # 🌟 命名修改：模型名称加上 _with_lai 标识
            model_path = os.path.join(MODEL_SAVE_DIR, f'bp_ann_model_{target}_with_lai_{i}.pkl')
            joblib.dump(model, model_path)
            print(f" 完成！")

        print(f"\n  📈 计算 {target.upper()} 强约束集成模型 (中位数) 的最终性能...")
        ensemble_predictions = np.column_stack(ensemble_predictions)
        median_pred = np.median(ensemble_predictions, axis=1)

        rmse = np.sqrt(mean_squared_error(y_val, median_pred))
        r2 = r2_score(y_val, median_pred)

        print(f"  🌟 {target.upper()} 最终集成性能指标 (19维输入): RMSE = {rmse:.4f}, R2 = {r2:.4f}\n")

        # 绘制验证图
        plot_validation(y_val, median_pred, target, r2, rmse)


def plot_validation(y_true, y_pred, target_name, r2, rmse):
    """绘制验证散点图"""
    plt.figure(figsize=(6, 6))

    if len(y_true) > 5000:
        hb = plt.hexbin(y_true, y_pred, gridsize=50, cmap='viridis', mincnt=1, edgecolors='none')
        cb = plt.colorbar(hb, label='Point Density (Count)')
    else:
        plt.scatter(y_true, y_pred, alpha=0.7, s=20, c='#1f77b4', edgecolors='white', linewidth=0.5)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')

    plt.xlabel(f'True {target_name.upper()}')
    plt.ylabel(f'Ensemble Median {target_name.upper()} (with LAI prior)')
    plt.title(f'{target_name.upper()} Validation (18 Kernels + LAI)\n$R^2$={r2:.3f}, RMSE={rmse:.3f}')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(MODEL_SAVE_DIR, f'validation_plot_{target_name}_with_lai_ensemble.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  🖼️ 集成验证图表已保存: {save_path}\n")


if __name__ == "__main__":
    train_inversion_models()