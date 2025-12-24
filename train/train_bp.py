import os
import sys
import csv
import glob
import pickle
import numpy as np
import pyabf
from scipy.signal import peak_widths
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. 配置部分
# -------------------------------------------------------------------

# 特征提取参数 (必须与 algorithms.py 保持一致)
SAMPLE_POINTS = 91  # 插值点数

# -------------------------------------------------------------------
# 2. 辅助函数
# -------------------------------------------------------------------

def extract_feature_segment(signal_data, peak_idx, sample_points=91):
    """
    从信号中提取峰值附近的波形片段作为特征
    逻辑参考 algorithms.py 中的 extract_features_from_abf
    """
    # 使用 peak_widths 的逻辑来确定截取范围 (这里简化处理，模拟 rel_height=0.5)
    # 注意：为了准确性，最好是根据 CSV 中记录的 width 或直接使用固定的窗口
    # 这里我们尝试动态计算宽度，模拟 algorithms.py 的行为
    
    # 临时反转信号用于计算宽度 (假设是负峰)
    inverted_signal = -signal_data
    
    try:
        # 计算单个峰的宽度信息
        results = peak_widths(inverted_signal, [peak_idx], rel_height=0.5)
        left = int(results[2][0])
        right = int(results[3][0])
        
        # 边界保护
        left = max(0, left)
        right = min(len(signal_data), right)
        
        if right - left < 5:
            return None
            
        raw_segment = signal_data[left:right]
        
        # 线性插值到固定长度
        interpolated = np.interp(
            np.linspace(0, len(raw_segment) - 1, sample_points),
            np.arange(len(raw_segment)),
            raw_segment
        )
        return interpolated
        
    except Exception as e:
        # print(f"特征提取失败 idx={peak_idx}: {e}")
        return None

def load_data_from_csv_and_abf(csv_path, abf_files=None):
    """
    读取 CSV 文件，解析峰值索引和标签，并从对应的 ABF 文件中提取波形特征
    """
    X = []
    y = []
    
    print(f"正在处理结果文件: {csv_path}")
    
    abf_path = None
    
    # 1. 直接使用提供的 abf_files 列表中的第一个文件
    # 这里的假设是：train.py 已经处理好了用户的选择逻辑
    # 如果用户选了多个，我们可能需要一个更复杂的逻辑来决定 CSV 对应哪个 ABF
    # 但根据最新需求：“去掉这个所谓的智能兜底的办法，兼容性保留也去掉，只留下从UI界面导入的”
    # 以及 “如果UI界面有导入两个abf文件，要运行那个我鼠标选择的”
    # 这意味着 train.py 应该只传过来用户选中的那个 ABF。
    
    if abf_files:
        # 简单粗暴：直接拿第一个。因为 train.py 已经保证 abf_files 里装的就是用户选的那个。
        # 如果用户在 UI 里选了多个 ABF 配合一个 CSV？通常是一对一。
        # 如果是多对多，逻辑会很复杂。这里假设用户一次训练只针对一组 (CSV + 对应 ABF)。
        # 或者 train.py 传递的是所有选中的 ABF，我们需要在这里挑一个？
        # 不，用户说“运行那个我鼠标选择的”，这通常意味着一次只能选一个主 ABF，或者选中的 ABF 都是有效的。
        # 但 CSV 里的 Peak_Index 是相对于某个特定 ABF 的。
        # 如果用户选了 ABF_1 和 ABF_2，CSV 是基于 ABF_1 的，那我们怎么知道用哪个？
        # 除非文件名匹配。但用户又说“去掉智能兜底...只留下从UI界面导入的”。
        # 我们可以保留最基础的文件名匹配（如果列表有多个），或者如果列表只有一个，就强制用那一个。
        
        # 策略：
        # 1. 如果 abf_files 只有一个，直接用，不检查文件名 (满足“运行那个我鼠标选择的”且“去掉智能兜底”)
        # 2. 如果 abf_files 有多个，尝试文件名匹配。如果匹配不到，报错（因为不知道用哪个）。
        
        if len(abf_files) == 1:
            abf_path = abf_files[0]
            # print(f"直接使用选中的 ABF 文件: {os.path.basename(abf_path)}")
        else:
            # 多个文件时，必须匹配文件名，否则无法对应
            csv_basename = os.path.basename(csv_path)
            core_name = csv_basename.replace("_classification_results.csv", "").replace(".csv", "")
            
            for f in abf_files:
                f_basename = os.path.basename(f)
                f_core = os.path.splitext(f_basename)[0]
                if f_core == core_name or f_core in csv_basename:
                    abf_path = f
                    break
            
            if not abf_path:
                print(f"警告: 传入了多个 ABF 文件，但没有一个与 CSV ({csv_basename}) 文件名匹配。无法确定使用哪个。")
                return [], []

    # 2. 如果没找到，直接返回空 (去掉了所有回退逻辑)
    if not abf_path:
        print(f"错误: 未找到对应的 ABF 文件。请确保在 UI 中选择了正确的 ABF 文件。")
        return [], []
        
    print(f"使用 ABF 文件: {abf_path}")
    
    try:
        abf = pyabf.ABF(abf_path)
        signal_data = abf.sweepY
    except Exception as e:
        print(f"读取 ABF 失败: {e}")
        return [], []

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # 获取关键信息
                peak_idx = int(row['Peak_Index'])
                label = row['Cluster_ID'] # 或者使用 Category_Name
                
                # 提取特征
                feature = extract_feature_segment(signal_data, peak_idx, SAMPLE_POINTS)
                
                if feature is not None:
                    X.append(feature)
                    y.append(label)
                    
            except KeyError:
                print("CSV 格式不匹配，缺少 Peak_Index 或 Cluster_ID")
                break
            except Exception as e:
                print(f"处理行失败: {e}")
                continue
                
    return X, y

# -------------------------------------------------------------------
# 3. 主训练流程
# -------------------------------------------------------------------

def train_bp_model(data_files, params, callback=None):
    """
    封装好的 BP 训练函数
    :param data_files: list of str, CSV 文件路径列表
    :param params: dict, 训练参数 (hidden_layers, lr, epochs, etc.)
    :param callback: function, 用于回传日志/进度的回调函数 (msg) -> None
    :return: dict, 训练结果 (model, score, loss_curve)
    """
    def log(msg):
        if callback:
            callback(msg)
        else:
            print(msg)

    log("=== 开始构建 BP 神经网络训练流程 ===")
    
    all_X = []
    all_y = []
    
    # 获取 ABF 文件列表 (如果通过 params 传递)
    abf_files = params.get('abf_files', None)
    
    # 2. 加载数据
    for csv_file in data_files:
        log(f"正在处理: {os.path.basename(csv_file)}")
        X, y = load_data_from_csv_and_abf(csv_file, abf_files)
        if X:
            all_X.extend(X)
            all_y.extend(y)
            
    if not all_X:
        log("没有提取到有效的训练数据。")
        return None
        
    X_arr = np.array(all_X)
    y_arr = np.array(all_y)
    
    log(f"数据集构建完成: 样本数={len(X_arr)}, 特征维度={X_arr.shape[1]}")
    log(f"类别标签: {np.unique(y_arr)}")
    
    # 3. 数据预处理
    X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. 构建 BP 神经网络
    hidden_layers = params.get('hidden_layers', (100, 50))
    lr = params.get('lr', 0.001)
    epochs = params.get('epochs', 500)
    batch_size = params.get('batch_size', 32)
    
    n_samples = X_train_scaled.shape[0]
    
    # 修复 batch_size 警告：如果 batch_size > 样本数，强制设为样本数
    if batch_size > n_samples:
        log(f"提示: 批次大小 ({batch_size}) 大于训练样本数 ({n_samples})，已自动调整为 {n_samples}")
        batch_size = n_samples
        
    # 使用 partial_fit 手动控制训练循环，以便监控准确率
    clf = MLPClassifier(hidden_layer_sizes=hidden_layers, 
                        learning_rate_init=lr,
                        batch_size=batch_size,
                        random_state=42,
                        max_iter=1, # 仅占位
                        warm_start=False) # 必须为 False，因为我们使用 partial_fit 且不需要 fit 的增量特性
    
    classes = np.unique(y_arr)
    loss_curve = []
    accuracy_curve = []
    
    log(f"开始训练 BP 网络 (Epochs: {epochs})...")
    
    n_batches = int(np.ceil(n_samples / batch_size))
    
    for epoch in range(epochs):
        # Shuffle
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train_scaled[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss_sum = 0
        
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_samples)
            batch_X = X_shuffled[start:end]
            batch_y = y_shuffled[start:end]
            
            clf.partial_fit(batch_X, batch_y, classes=classes)
            epoch_loss_sum += clf.loss_
            
        # 记录每个 Epoch 的指标 (使用平均 Loss)
        avg_loss = epoch_loss_sum / n_batches if n_batches > 0 else 0
        train_acc = clf.score(X_train_scaled, y_train)
        
        loss_curve.append(avg_loss)
        accuracy_curve.append(train_acc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Acc: {train_acc*100:.2f}%")

    # 5. 评估
    log("=== 模型评估 ===")
    y_pred = clf.predict(X_test_scaled)
    report = classification_report(y_test, y_pred)
    log(report)
    
    # 6. 保存与返回
    result = {
        'model': clf,
        'scaler': scaler,
        'feature_len': SAMPLE_POINTS,
        'classes': classes,
        'loss_curve': loss_curve,
        'accuracy_curve': accuracy_curve,
        'report': report
    }
    
    # try:
    #     with open(MODEL_SAVE_PATH, 'wb') as f:
    #         pickle.dump(result, f)
    #     log(f"模型已保存至: {MODEL_SAVE_PATH}")
    # except Exception as e:
    #     log(f"模型保存失败: {e}")
        
    return result

def main():
    # 兼容旧的 main 函数逻辑
    if not os.path.exists(DATA_RESULT_DIR):
        print(f"错误: 数据目录不存在 {DATA_RESULT_DIR}")
        return

    csv_pattern = os.path.join(DATA_RESULT_DIR, "**", "classification_results.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    if not csv_files:
        print("未找到任何 classification_results.csv 文件。")
        return

    params = {
        'hidden_layers': (100, 50),
        'lr': 0.001,
        'epochs': 500
    }
    
    result = train_bp_model(csv_files, params)
    
    if result:
        # 保存模型
        model_package = {
            'model': result['model'],
            'scaler': result['scaler'],
            'feature_len': result['feature_len'],
            'classes': result['classes']
        }
        with open(MODEL_SAVE_PATH, 'wb') as f:
            pickle.dump(model_package, f)
        print(f"\n模型已保存至: {MODEL_SAVE_PATH}")
        
        # 绘制 Loss
        if 'loss_curve' in result:
            plt.figure(figsize=(8, 5))
            plt.plot(result['loss_curve'])
            plt.title('Training Loss Curve')
            plt.savefig(os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'training_loss.png'))