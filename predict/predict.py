# predict.py
import os
import csv
import time
import numpy as np
import traceback
from PyQt5.QtWidgets import QWidget, QMessageBox, QProgressDialog, QFileDialog, QApplication
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure

from .UI import PredictUI, get_chinese_font
from .algorithms import Algorithms  # 确保引用路径正确


class PredictPage(QWidget):
    """推测页面逻辑"""

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.ui = PredictUI(self)

        # 状态数据
        self.current_results = None  # 存储预测结果: {'path':..., 'peaks':..., 'labels':..., 'label_map':...}

        # 信号连接
        self.ui.run_btn.clicked.connect(self.run_inference)
        self.ui.save_btn.clicked.connect(self.save_inference_results)

        # 数据同步
        self.data_manager.models_changed_signal.connect(self.refresh_lists)
        self.data_manager.submissions_changed_signal.connect(self.refresh_lists)
        self.refresh_lists()

    def refresh_lists(self):
        """刷新模型列表和数据列表"""
        self.ui.model_list.clear()
        self.ui.model_list.addItems(self.data_manager.get_model_names())

        self.ui.data_list.clear()
        self.ui.data_list.addItems(self.data_manager.get_submission_names())

    def _get_inference_colors(self, n_clusters):
        """获取颜色列表，保持与 ClusteringPage 一致"""
        custom_colors = ['#FF0000', '#FFA500', '#FFFF00', '#008000',
                         '#00BFFF', '#00FFFF', '#800080', '#FFC0CB',
                         '#A52A2A', '#FFD700', '#00FF7F', '#7B68EE',
                         '#C0C0C0', '#000000', '#FFF8DC', '#808000']
        return [custom_colors[i % len(custom_colors)] for i in range(n_clusters)]

    def run_inference(self):
        """执行推测"""
        model_item = self.ui.model_list.currentItem()
        if not model_item:
            QMessageBox.warning(self, "提示", "请先选择一个模型。")
            return
        model_info = self.data_manager.get_model_by_name(model_item.text())
        if not model_info:
            QMessageBox.critical(self, "错误", "模型数据损坏。")
            return
            
        # 兼容性检查：如果是 KMeans 模型，确保 model_obj 存在
        # 如果是 BP/SNN 模型，它们没有 model_obj，所以不能直接检查 model_obj
        # 这里移除对 model_obj 的强制检查，改为在下方分支判断


        data_item = self.ui.data_list.currentItem()
        if not data_item:
            QMessageBox.warning(self, "提示", "请选择待分析的数据。")
            return

        data_tuple = self.data_manager.get_data_by_name(data_item.text())
        if not data_tuple: return

        path, peaks, _, _ = data_tuple

        try:
            print(f"开始推测: 模型={model_item.text()}, 数据={data_item.text()}")

            # 根据模型类型选择不同的推测逻辑
            if 'model_obj' in model_info:
                # --- 1. KMeans 模型 (来自 ClusteringPage) ---
                kmeans_model = model_info['model_obj']
                label_map = model_info.get('label_map', {})
                
                # 特征提取
                segments, signal_data, valid_indices = Algorithms.extract_features_from_abf(path, peaks)
                if len(segments) == 0:
                    QMessageBox.warning(self, "警告", "未提取到有效波形片段。")
                    return
                
                predicted_labels = kmeans_model.predict(segments)
                
            elif 'model' in model_info:
                # --- 2. BP 神经网络 (来自 TrainPage) ---
                signal_data, predicted_labels, valid_indices = Algorithms.run_bp_inference(path, peaks, model_info)
                if len(predicted_labels) == 0:
                    QMessageBox.warning(self, "警告", "未提取到有效波形片段。")
                    return
                
                # 构造 label_map (BP 直接输出类别值，无论是字符串还是数字)
                unique_labels = np.unique(predicted_labels)
                label_map = {lbl: str(lbl) for lbl in unique_labels}
                
            elif 'model_state_dict' in model_info:
                # --- 3. SNN 神经网络 (来自 TrainPage) ---
                signal_data, predicted_labels, valid_indices = Algorithms.run_snn_inference(path, peaks, model_info)
                if len(predicted_labels) == 0:
                    QMessageBox.warning(self, "警告", "未提取到有效波形片段。")
                    return
                    
                # 构造 label_map
                unique_labels = np.unique(predicted_labels)
                label_map = {lbl: str(lbl) for lbl in unique_labels}
                
            else:
                QMessageBox.critical(self, "错误", "未知的模型格式，无法进行推测。")
                return

            valid_peaks = peaks[valid_indices]

            self.current_results = {
                'path': path,
                'signal_data': signal_data,
                'peaks': valid_peaks,
                'labels': predicted_labels,
                'label_map': label_map,
                'model_name': model_item.text(),
                'data_name': data_item.text()
            }

            # 绘图展示 (这里会调用修改后的 plot_results)
            self.plot_results(signal_data, valid_peaks, predicted_labels, label_map)
            QMessageBox.information(self, "完成", "推测完成！结果已显示。")

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "推测失败", str(e))

    def plot_results(self, signal_data, peak_indices, cluster_labels, label_map):
        """
        绘制推测结果
        使用 Min-Max 降采样策略，确保波形与峰值对齐
        """
        self.ui.clear_plots()

        print("Inference: 正在绘制主图...")

        # 1. 创建 Figure (UI.add_plot 会将其转换为 Canvas)
        # 增加高度以获得更好的视野
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        # ---------------------------------------------------------
        # 核心逻辑：Min-Max 降采样 (完全复用 ClusteringPage 逻辑)
        # ---------------------------------------------------------
        target_points = 5000
        data_length = len(signal_data)
        sampling_step = max(1, data_length // target_points)
        num_intervals = data_length // sampling_step

        # 准备降采样数组
        x_display = np.zeros(num_intervals * 2)
        y_display = np.zeros(num_intervals * 2)

        valid_points = 0
        for i in range(num_intervals):
            start = i * sampling_step
            end = min(start + sampling_step, data_length)
            if start >= end:
                continue

            segment = signal_data[start:end]
            # 计算区间内的最大值和最小值
            y_min = np.nanmin(segment)
            y_max = np.nanmax(segment)

            # 构造 x 坐标 (使用区间的起始点)
            x_val = start

            idx = valid_points * 2
            x_display[idx] = x_val
            y_display[idx] = y_min
            x_display[idx + 1] = x_val
            y_display[idx + 1] = y_max
            valid_points += 1

        x_display = x_display[:valid_points * 2]
        y_display = y_display[:valid_points * 2]

        # 2. 绘制降采样后的信号背景
        ax.plot(x_display, y_display, label='原始信号', color='blue', alpha=0.6, zorder=1, linewidth=0.8)

        # 3. 绘制分类后的峰值点
        unique_labels = sorted(list(np.unique(cluster_labels)))
        colors = self._get_inference_colors(len(unique_labels))

        # 这里的 map 用于颜色索引
        color_map = {lbl: i for i, lbl in enumerate(unique_labels)}

        # 为了图例整洁，我们按类别分组绘制
        for lbl in unique_labels:
            # 获取该类别的所有索引
            indices = (cluster_labels == lbl)
            current_peaks = peak_indices[indices]
            current_amps = signal_data[current_peaks]

            # 获取人类可读的标签名称
            lbl_name = label_map.get(lbl, f"Cluster {lbl}")
            c = colors[color_map[lbl] % len(colors)]

            # 绘制散点
            ax.scatter(current_peaks, current_amps, c=c, label=lbl_name, s=25, zorder=2, edgecolors='white',
                       linewidth=0.5)

        # 4. 设置标题和样式
        ax.set_title(f"推测结果: {self.current_results.get('data_name', '')}", fontproperties=get_chinese_font(),
                     fontsize=16)
        ax.set_xlabel("时间点", fontproperties=get_chinese_font())
        ax.set_ylabel("幅值", fontproperties=get_chinese_font())
        ax.legend(prop=get_chinese_font(), fontsize=12, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)

        # 添加主图到 UI
        self.ui.add_plot(fig)

        # ---------------------------------------------------------
        # 附带绘制：类别统计柱状图
        # ---------------------------------------------------------
        fig_bar = Figure(figsize=(6, 4), dpi=100)
        ax_bar = fig_bar.add_subplot(111)

        counts = [np.sum(cluster_labels == lbl) for lbl in unique_labels]
        names = [label_map.get(lbl, str(lbl)) for lbl in unique_labels]

        # 使用对应的颜色
        bar_colors = [colors[color_map[lbl] % len(colors)] for lbl in unique_labels]

        bars = ax_bar.bar(names, counts, color=bar_colors, alpha=0.8)
        ax_bar.set_title("分类数量统计", fontproperties=get_chinese_font())
        ax_bar.grid(True, axis='y', linestyle='--', alpha=0.3)

        # 在柱子上显示数字
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')

        self.ui.add_plot(fig_bar)

    def save_inference_results(self):
        """保存：CSV统计 + 每一张峰的图片(带标签)"""
        if not self.current_results:
            QMessageBox.warning(self, "提示", "请先运行推测。")
            return

        save_dir = QFileDialog.getExistingDirectory(self, "选择保存结果的文件夹")
        if not save_dir: return

        # 创建文件夹
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder_name = f"Inference_{self.current_results['data_name']}_by_{self.current_results['model_name']}_{timestamp}"
        out_path = os.path.join(save_dir, folder_name)
        img_path = os.path.join(out_path, "classified_images")

        os.makedirs(out_path, exist_ok=True)
        os.makedirs(img_path, exist_ok=True)

        # 准备数据
        peaks = self.current_results['peaks']
        labels = self.current_results['labels']
        label_map = self.current_results['label_map']
        full_signal = self.current_results['signal_data']
        full_x = np.arange(len(full_signal))

        total = len(peaks)

        # 进度条
        progress = QProgressDialog("正在保存分类结果...", "取消", 0, total, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        try:
            # 1. 保存 CSV
            csv_file = open(os.path.join(out_path, "classification_results.csv"), 'w', newline='', encoding='utf-8-sig')
            writer = csv.writer(csv_file)
            writer.writerow(["Index", "Peak_Index", "Amplitude", "Cluster_ID", "Category_Name"])

            # 2. 循环保存图片和CSV
            for i, (p_idx, lbl) in enumerate(zip(peaks, labels)):
                if progress.wasCanceled(): break

                cat_name = label_map.get(lbl, f"Cluster {lbl}")
                amp = full_signal[p_idx]

                writer.writerow([i + 1, p_idx, amp, lbl, cat_name])

                # 绘制单峰图
                fig = Figure(figsize=(4, 3), dpi=80)
                ax = fig.add_subplot(111)

                window = 100
                s = max(0, p_idx - window)
                e = min(len(full_signal), p_idx + window)

                ax.plot(full_x[s:e], full_signal[s:e], color='blue')
                ax.scatter([p_idx], [amp], c='red', s=30, zorder=5)  # 确保点在最上层

                ax.set_title(f"Peak {i + 1}: {cat_name}", color='red', fontweight='bold',
                             fontproperties=get_chinese_font())

                # 安全文件名
                safe_cat_name = str(cat_name).replace("/", "_").replace("\\", "_")
                fname = f"ID{i + 1:04d}_{safe_cat_name}_idx{p_idx}.png"
                fig.savefig(os.path.join(img_path, fname))
                fig.clear()

                progress.setValue(i + 1)
                QApplication.processEvents()

            csv_file.close()
            progress.setValue(total)
            QMessageBox.information(self, "成功", f"结果已保存至:\n{out_path}")

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "保存失败", str(e))