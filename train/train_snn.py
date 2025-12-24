import os
import csv
import glob
import pickle
import numpy as np
import pyabf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from scipy.signal import peak_widths
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. 配置部分
# -------------------------------------------------------------------

# DATA_RESULT_DIR = "/Users/asuka/Desktop/Inference_1_by_kmeans_20251204_000415"

# 对应的原始 ABF 文件路径 (如果 CSV 里没有记录完整路径，需要在这里指定搜索目录)
# ABF_FILE_DIR = "/Users/asuka/Desktop/Inference_1_by_kmeans_20251204_000415"

# 模型保存路径
# MODEL_SAVE_PATH = "/Users/asuka/Documents/Program/pythonProject/Nanopores/Nanopores_tool_new/train/snn_model.pkl"
SAMPLE_POINTS = 91

# SNN 参数
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
TIME_STEPS = 20  # 模拟的时间步长
TAU = 2.0        # 膜电位时间常数

# -------------------------------------------------------------------
# 2. SNN 核心组件 (手写实现，无需额外库)
# -------------------------------------------------------------------

class SurrogateHeaviside(torch.autograd.Function):
    """
    替代梯度函数：在前向传播时表现为阶跃函数 (Heaviside)，
    在反向传播时使用平滑函数的导数 (如 fast sigmoid) 来近似，
    从而允许梯度流过脉冲发射过程。
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # 使用 Fast Sigmoid 的导数作为近似: f'(x) = 1 / (1 + |x|)^2
        grad_input = grad_output / (1 + torch.abs(input))**2
        return grad_input

class LIFNode(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) 神经元
    """
    def __init__(self, tau=2.0, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.act = SurrogateHeaviside.apply

    def forward(self, x, v):
        # LIF 差分方程: v[t] = v[t-1] + (I[t] - (v[t-1] - v_reset)) / tau
        # 这里输入 x 相当于电流 I
        v = v + (x - (v - self.v_reset)) / self.tau
        
        # 产生脉冲
        spike = self.act(v - self.v_threshold)
        
        # 发生脉冲后重置膜电位 (Soft Reset: 减去阈值，Hard Reset: 设为 v_reset)
        # 这里使用 Hard Reset 的变体 (v_reset通常为0)
        v = v * (1 - spike) + self.v_reset * spike
        
        return spike, v

class SNN(nn.Module):
    """
    全连接 SNN 模型
    结构: Input -> Linear -> LIF -> Linear -> LIF (Output)
    """
    def __init__(self, input_size, hidden_size, output_size, num_steps=20):
        super().__init__()
        self.num_steps = num_steps
        
        # 第一层: 编码/隐藏层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = LIFNode(tau=TAU)
        
        # 第二层: 输出层
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = LIFNode(tau=TAU)

    def forward(self, x):
        # x shape: [batch_size, input_size]
        # 我们将静态输入 x 重复 num_steps 次作为恒定电流输入 (Rate Coding 的一种形式)
        
        batch_size = x.shape[0]
        
        # 初始化膜电位
        v1 = torch.zeros(batch_size, self.fc1.out_features, device=x.device)
        v2 = torch.zeros(batch_size, self.fc2.out_features, device=x.device)
        
        # 记录输出层的脉冲总数 (Rate Coding 输出)
        spk2_sum = torch.zeros(batch_size, self.fc2.out_features, device=x.device)
        
        # 时间步循环
        for t in range(self.num_steps):
            # Layer 1
            cur1 = self.fc1(x) # 恒定电流输入
            spk1, v1 = self.lif1(cur1, v1)
            
            # Layer 2
            cur2 = self.fc2(spk1) # 上一层的脉冲作为这一层的电流输入
            spk2, v2 = self.lif2(cur2, v2)
            
            # 累加输出脉冲
            spk2_sum += spk2
            
        # 返回输出层的总脉冲率作为 logits/probabilities 的替代
        return spk2_sum / self.num_steps

# -------------------------------------------------------------------
# 3. 数据加载 (复用部分)
# -------------------------------------------------------------------

class WaveformDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def extract_feature_segment(signal_data, peak_idx, sample_points=91):
    inverted_signal = -signal_data
    try:
        results = peak_widths(inverted_signal, [peak_idx], rel_height=0.5)
        left = int(results[2][0])
        right = int(results[3][0])
        left = max(0, left)
        right = min(len(signal_data), right)
        
        if right - left < 5:
            return None
            
        raw_segment = signal_data[left:right]
        interpolated = np.interp(
            np.linspace(0, len(raw_segment) - 1, sample_points),
            np.arange(len(raw_segment)),
            raw_segment
        )
        return interpolated
    except:
        return None

def load_data_from_csv_and_abf(csv_path, abf_files=None):
    X = []
    y = []
    
    abf_path = None
    
    # 1. 直接使用提供的 abf_files 列表中的第一个文件
    # 逻辑同 train_bp.py: 只使用 UI 传过来的文件，去掉所有自动搜寻和回退
    
    if abf_files:
        if len(abf_files) == 1:
            abf_path = abf_files[0]
        else:
            # 多个文件时，必须匹配文件名
            csv_basename = os.path.basename(csv_path)
            core_name = csv_basename.replace("_classification_results.csv", "").replace(".csv", "")
            
            for f in abf_files:
                f_basename = os.path.basename(f)
                f_core = os.path.splitext(f_basename)[0]
                if f_core == core_name or f_core in csv_basename:
                    abf_path = f
                    break
            
            if not abf_path:
                # print(f"警告: 传入了多个 ABF 文件，但没有一个与 CSV ({csv_basename}) 文件名匹配。")
                return [], []

    # 2. 如果没找到，直接返回空 (去掉了所有回退逻辑)
    if not abf_path:
        # print(f"错误: 未找到对应的 ABF 文件。请确保在 UI 中选择了正确的 ABF 文件。")
        return [], []
        
    try:
        abf = pyabf.ABF(abf_path)
        signal_data = abf.sweepY
    except:
        return [], []

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                peak_idx = int(row['Peak_Index'])
                label = row['Cluster_ID'] # 使用 ID 作为标签
                feature = extract_feature_segment(signal_data, peak_idx, SAMPLE_POINTS)
                if feature is not None:
                    X.append(feature)
                    y.append(label)
            except:
                continue
    return X, y

# -------------------------------------------------------------------
# 4. 主流程
# -------------------------------------------------------------------

def train_snn_model(data_files, params, callback=None):
    """
    封装好的 SNN 训练函数
    """
    def log(msg):
        if callback:
            callback(msg)
        else:
            print(msg)
            
    log("=== SNN (Spiking Neural Network) 训练流程 ===")
    
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
        log("未加载到数据。")
        return None
        
    # 2. 预处理
    X_arr = np.array(all_X)
    y_arr = np.array(all_y)
    
    # Label Encoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_arr)
    num_classes = len(label_encoder.classes_)
    
    log(f"数据集: {len(X_arr)} 样本, {num_classes} 类别")
    
    X_train, X_test, y_train, y_test = train_test_split(X_arr, y_encoded, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SNN 参数
    batch_size = params.get('batch_size', 32)
    lr = params.get('lr', 1e-3)
    num_epochs = params.get('epochs', 100)
    time_steps = params.get('time_steps', 20)
    tau = params.get('tau', 2.0)
    
    train_dataset = WaveformDataset(X_train_scaled, y_train)
    test_dataset = WaveformDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"使用设备: {device}")
    
    model = SNN(input_size=SAMPLE_POINTS, 
                hidden_size=100, 
                output_size=num_classes, 
                num_steps=time_steps).to(device)
                
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. 训练循环
    log(f"开始训练 (Epochs: {num_epochs})...")
    loss_history = []
    accuracy_history = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward
            outputs = model(inputs) 
            
            # Loss
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = epoch_loss / len(train_loader)
        acc = 100 * correct / total
        loss_history.append(avg_loss)
        accuracy_history.append(acc/100.0) # 归一化到 0-1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
            
    # 5. 评估
    log("=== 模型评估 ===")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    report = classification_report(all_labels, all_preds, target_names=[str(c) for c in label_encoder.classes_])
    log(report)
    
    # 6. 返回结果
    result = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'label_encoder': label_encoder,
        'config': {
            'input_size': SAMPLE_POINTS,
            'hidden_size': 100,
            'output_size': num_classes,
            'num_steps': time_steps,
            'tau': tau
        },
        'loss_curve': loss_history,
        'accuracy_curve': accuracy_history,
        'report': report
    }
    return result

def main():
    print("=== SNN (Spiking Neural Network) 训练流程 ===")
    
    # 1. 准备数据
    if not os.path.exists(DATA_RESULT_DIR):
        print(f"请设置正确的数据路径: {DATA_RESULT_DIR}")
        return

    csv_pattern = os.path.join(DATA_RESULT_DIR, "**", "classification_results.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    if not csv_files:
        print("未找到任何 classification_results.csv 文件。")
        return
        
    params = {
        'epochs': 100,
        'lr': 1e-3
    }
    
    result = train_snn_model(csv_files, params)
    
    if result:
        torch.save(result, MODEL_SAVE_PATH)
        print(f"模型已保存至: {MODEL_SAVE_PATH}")
        
        # 绘制 Loss
        plt.figure(figsize=(8, 5))
        plt.plot(result['loss_curve'])
        plt.title('SNN Training Loss')
        plt.savefig(os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'snn_loss.png'))
