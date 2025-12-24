# algorithms.py
import numpy as np
import pyabf
import torch
from scipy.signal import peak_widths
from sklearn.cluster import KMeans
from train.train_snn import SNN, LIFNode, SurrogateHeaviside

class Algorithms:

    @staticmethod
    def extract_features_from_abf(path, peak_indices, n_samples=91):
        """
        静态辅助方法：从ABF文件和峰值索引提取统一的特征（插值后的波形片段）。
        用于训练和推理的一致性。
        """
        abf = pyabf.ABF(path)
        signal_data = abf.sweepY

        segments = []
        # 使用 peak_widths 确定截取范围，这里逻辑需与原逻辑保持一致
        # 注意：这里假设 peak_indices 是基于 signal_data 的
        results = peak_widths(-signal_data, peak_indices, rel_height=0.5)

        valid_indices = []  # 记录有效的索引（有些可能因为太窄被过滤）

        for i, peak in enumerate(peak_indices):
            left = int(results[2][i])
            right = int(results[3][i])

            # 边界检查
            left = max(0, left)
            right = min(len(signal_data), right)

            if right - left < 5:
                continue

            raw_segment = signal_data[left:right]

            # 线性插值到固定长度
            interpolated = np.interp(
                np.linspace(0, len(raw_segment) - 1, n_samples),
                np.arange(len(raw_segment)),
                raw_segment
            )
            segments.append(interpolated)
            valid_indices.append(i)  # 记录对应原始 peak_indices 的第几个

        return np.array(segments), signal_data, valid_indices

    @staticmethod
    def run_kmeans(path, peak_indices, params):
        """
        运行 KMeans 并返回 (signal_data, labels, model, valid_indices)
        """
        # 1. 提取特征
        segments, signal_data, valid_indices = Algorithms.extract_features_from_abf(path, peak_indices)

        if len(segments) == 0:
            return signal_data, [], None, []

        # 2. 训练模型
        kmeans = KMeans(
            n_clusters=params['n_clusters'],
            n_init=params['n_init'],
            max_iter=params['max_iter'],
            random_state=params['random_state']
        ).fit(segments)

        labels = kmeans.labels_

        # 返回: 原始信号, 聚类标签, 训练好的模型对象, 有效的索引列表
        return signal_data, labels, kmeans, valid_indices

    @staticmethod
    def run_dbscan(data, params):
        # DBSCAN 是非参数模型，通常很难直接用于新数据的“预测”，
        # 除非使用 KNN 寻找最近邻。这里暂时保持原样或仅返回标签。
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
        labels = dbscan.fit_predict(data)
        return labels, None  # DBSCAN 没有 predict 方法，返回 None 模型

    @staticmethod
    def run_bp_inference(path, peak_indices, model_info):
        """
        BP 神经网络推测
        """
        # 1. 提取特征
        segments, signal_data, valid_indices = Algorithms.extract_features_from_abf(path, peak_indices)
        if len(segments) == 0:
            return signal_data, [], []

        # 2. 预处理 (Scaler)
        if 'scaler' in model_info:
            scaler = model_info['scaler']
            segments = scaler.transform(segments)
        
        # 3. 预测
        clf = model_info['model']
        labels = clf.predict(segments)
        
        return signal_data, labels, valid_indices

    @staticmethod
    def run_snn_inference(path, peak_indices, model_info):
        """
        SNN 神经网络推测
        """
        # 1. 提取特征
        segments, signal_data, valid_indices = Algorithms.extract_features_from_abf(path, peak_indices)
        if len(segments) == 0:
            return signal_data, [], []

        # 2. 预处理
        if 'scaler' in model_info:
            scaler = model_info['scaler']
            segments = scaler.transform(segments)
        
        # 3. 加载模型
        config = model_info['config']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 重建模型结构
        model = SNN(input_size=config['input_size'], 
                    hidden_size=config['hidden_size'], 
                    output_size=config['output_size'], 
                    num_steps=config['num_steps']).to(device)
        
        # 加载参数
        model.load_state_dict(model_info['model_state_dict'])
        model.eval()
        
        # 4. 预测
        X_tensor = torch.FloatTensor(segments).to(device)
        
        with torch.no_grad():
            outputs = model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            
        pred_indices = predicted.cpu().numpy()
        
        # 5. 解码标签
        if 'label_encoder' in model_info:
            le = model_info['label_encoder']
            labels = le.inverse_transform(pred_indices)
        else:
            labels = pred_indices
            
        return signal_data, labels, valid_indices