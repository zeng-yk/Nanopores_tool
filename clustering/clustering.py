"""
Clustering 模块的主要功能逻辑实现
"""
import os
import re
import time
import pickle
import traceback
import numpy as np
from PyQt5.QtWidgets import QWidget, QMessageBox, QProgressDialog, QFileDialog, QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QInputDialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from .UI import ClusteringUI, get_chinese_font
from clustering.algorithms import Algorithms
from parameter_widgets import BaseParameterWidget


# 定义一个简单的对话框类，用于给每个 Cluster 命名
class ClusterLabelDialog(QDialog):
    def __init__(self, n_clusters, parent=None):
        super().__init__(parent)
        self.setWindowTitle("定义类别标签")
        self.layout = QFormLayout(self)
        self.inputs = {}

        for i in range(n_clusters):
            le = QLineEdit(f"Cluster {i}")
            self.inputs[i] = le
            self.layout.addRow(f"类别 {i} 名称:", le)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addRow(self.buttons)

    def get_labels(self):
        """返回 {0: '名字', 1: '名字'}"""
        mapping = {}
        for i, le in self.inputs.items():
            text = le.text().strip()
            mapping[i] = text if text else f"Cluster {i}"
        return mapping


class ClusteringPage(QWidget):
    """聚类页面的主要逻辑实现类"""

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)

        # 存储数据管理器引用
        self.data_manager = data_manager

        # 初始化结果变量
        self.results = None
        self.signal_for_plot = None
        self.peaks_for_plot = None
        self.labels_for_plot = None
        self.features_for_plot = None

        self.current_model = None  # 新增：用于临时存储当前训练好的模型

        # 初始化UI
        self.ui = ClusteringUI(self)

        # 连接信号和槽
        self.connect_signals()

        # 更新提交列表
        self.data_manager.submissions_changed_signal.connect(self.update_submission_list)
        self.update_submission_list()  # 初始化时显式更新一次列表

    def connect_signals(self):
        """连接所有的信号槽"""
        # 下拉列表变化信号
        self.ui.function_selector.currentIndexChanged.connect(self.on_function_changed)

        # 按钮点击信号
        self.ui.button.clicked.connect(self.run_selected_algorithm)
        self.ui.save_button.clicked.connect(self.save_data)
        if hasattr(self.ui, 'save_model_button'):
            self.ui.save_model_button.clicked.connect(self.save_trained_model)

    def on_function_changed(self, index):
        """下拉列表选项改变时的处理函数"""
        selected_text = self.ui.function_selector.itemText(index)
        print(f"下拉列表切换到: {selected_text} (索引: {index})")

        # 更新参数设置区域
        if selected_text in self.ui.param_widgets:
            widget_to_show = self.ui.param_widgets[selected_text]
            self.ui.parameter_stack.setCurrentWidget(widget_to_show)
            print(f"参数区显示: {widget_to_show.__class__.__name__}")
        else:
            print(f"错误: 在 param_widgets 中未找到 '{selected_text}'")

    def update_submission_list(self):
        """刷新提交列表"""
        print("UI: 开始更新 submission 名称列表...")

        # 获取名称列表
        submission_names = self.data_manager.get_submission_names()

        # 清空列表并添加名称
        self.ui.submission_list_widget.clear()

        if submission_names:
            self.ui.submission_list_widget.addItems(submission_names)
            print(f"UI: QListWidget 已填充以下名称: {submission_names}")
        else:
            print("UI: DataManager 中没有 submission 名称可供显示。")

        print("UI: 列表更新完成。")

    def get_current_parameters(self) -> dict:
        """获取当前显示的参数控件的参数"""
        current_param_widget = self.ui.parameter_stack.currentWidget()
        if isinstance(current_param_widget, BaseParameterWidget):
            try:
                return current_param_widget.get_parameters()
            except NotImplementedError:
                print(f"错误: {current_param_widget.__class__.__name__} 未实现 get_parameters 方法")
                return None
        return None  # 如果当前页面不是参数控件

    def run_selected_algorithm(self):
        """获取参数，运行算法，并更新多个绘图"""
        selected_algorithm_name = self.ui.function_selector.currentText()
        if not selected_algorithm_name:
            print("错误：没有选中任何算法。")
            return
        print(f"选定算法: {selected_algorithm_name}")

        # 获取当前算法的参数
        parameters = self.get_current_parameters()
        if parameters is None:
            print("错误：无法获取当前算法的参数。")
            return
        print(f"获取参数: {parameters}")

        # 检查是否选择了数据项
        current_item = self.ui.submission_list_widget.currentItem()
        if not current_item:
            print("错误：请在'已提交的识别数据'中选择要进行聚类的数据。")
            QMessageBox.warning(self, "缺少数据", "请在'已提交的识别数据'中选择要进行聚类的数据项。")
            return

        selected_data_name = current_item.text()
        print(f"目标数据: {selected_data_name}")

        # 从数据管理器中获取数据
        path_to_process, peaks_to_process, result_width, prominences = self.data_manager.get_data_by_name(
            selected_data_name)

        widths = result_width[0]
        heights = result_width[1]

        if peaks_to_process is None:
            print(f"错误：无法从 DataManager 获取名为 '{selected_data_name}' 的数据。")
            QMessageBox.critical(self, "数据错误", f"无法找到或加载名为 '{selected_data_name}' 的数据。")
            return

        print(f"选定算法: {selected_algorithm_name}, 参数: {parameters}, 数据: {selected_data_name}")

        # 重置结果变量
        self.signal_for_plot = None
        self.peaks_for_plot = None
        self.labels_for_plot = None
        self.features_for_plot = None

        # 准备特征数据
        filtered_features_list = []  # 存储通过过滤的特征 [height, width, prominence]
        for i, peak_idx in enumerate(peaks_to_process):
            filtered_features_list.append([heights[i], widths[i], prominences[i]])

        features_array = np.array(filtered_features_list)
        self.features_for_plot = features_array

        try:
            if selected_algorithm_name == "K-Means":
                print("调用 KMeans 算法...")
                result_tuple = Algorithms.run_kmeans(path_to_process, peaks_to_process, parameters)
                if result_tuple and len(result_tuple) == 4:
                    self.signal_for_plot, labels_raw, self.current_model, valid_indices = result_tuple

                    # 因为插值过滤了部分短波形，我们需要对齐 labels 和 peaks
                    # valid_indices 是原始 peaks_to_process 中的下标
                    self.peaks_for_plot = peaks_to_process[valid_indices]
                    self.labels_for_plot = labels_raw

                    # 更新 features_for_plot 也要对其
                    self.features_for_plot = features_array[valid_indices]
                else:
                    print("警告：KMeans 算法未按预期返回信号、标签。可能无法绘制所有图形。")
                    self.labels_for_plot = None  # 标记为无效结果

            elif selected_algorithm_name == "DBSCAN":
                print("调用 DBSCAN 算法...")
                print("DBSCAN 的绘图更新逻辑尚未完全实现（需要返回特征）。")
                QMessageBox.information(self, "提示", "DBSCAN 结果绘图（带特征）暂未完全实现。")
                self.labels_for_plot = None

            elif selected_algorithm_name == "功能 3":
                print("功能 3 不需要执行聚类算法。")
                self._clear_plot_layout()  # 清除绘图区
                self.ui._display_plot_error("功能 3 无绘图结果。")  # 显示提示信息
                self.labels_for_plot = None  # 无有效标签
                return

            else:
                QMessageBox.warning(self, "未实现", f"算法 '{selected_algorithm_name}' 的执行逻辑尚未实现。")
                self.labels_for_plot = None
                return

            # 更新绘图区
            if self.labels_for_plot is not None and self.signal_for_plot is not None and self.peaks_for_plot is not None:
                print(f"算法执行成功！准备更新 {len(self.labels_for_plot)} 个峰值的绘图...")
                self.update_plots(self.signal_for_plot,
                                  self.peaks_for_plot,
                                  self.labels_for_plot,
                                  self.features_for_plot,
                                  "电流/NA")  # Y 轴标签
            elif self.labels_for_plot is None and selected_algorithm_name != "功能 3":
                print("算法执行完成，但没有返回有效的聚类结果或绘图所需数据。")
                self._clear_plot_layout()
                self.ui._display_plot_error("算法未返回有效结果\n或缺少绘图所需数据。")

        except ImportError:
            print("错误：未能导入 algorithms 模块或其中包含的算法库。")
            QMessageBox.critical(self, "导入错误", "运行算法所需的库未能导入，请检查安装。")
            self._clear_plot_layout()
            self.ui._display_plot_error("导入错误，无法运行算法。")
        except Exception as e:
            print(f"运行算法 '{selected_algorithm_name}' 时出错: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "算法错误", f"执行 '{selected_algorithm_name}' 时发生错误:\n{e}")
            self._clear_plot_layout()
            self.ui._display_plot_error(f"算法执行错误:\n{e}")

        print("--- 算法运行与绘图更新结束 ---")

    # 新增：保存模型的方法
    def save_trained_model(self):
        if self.current_model is None:
            QMessageBox.warning(self, "无模型", "请先运行 K-Means 算法生成模型。")
            return

        # 1. 输入模型名称
        name, ok = QInputDialog.getText(self, "保存模型", "请输入模型名称:")
        if not ok or not name.strip():
            return

        # 2. 弹出对话框，配置类别标签
        # 获取 cluster 数量
        n_clusters = self.current_model.n_clusters
        dialog = ClusterLabelDialog(n_clusters, self)
        if dialog.exec_() == QDialog.Accepted:
            label_map = dialog.get_labels()

            # 3. 保存到磁盘
            model_info = {
                'name': name.strip(),
                'type': 'KMeans',
                'model_obj': self.current_model,
                'label_map': label_map,
                'timestamp': time.time()
            }
            
            try:
                # 使用 DataManager 的 model_dir
                # 简单的文件名清洗，防止非法字符
                safe_name = "".join([c for c in name.strip() if c.isalnum() or c in (' ', '.', '_', '-')]).rstrip()
                if not safe_name:
                    safe_name = "model"
                
                filename = f"{safe_name}.pkl"
                save_path = os.path.join(self.data_manager.model_dir, filename)
                
                with open(save_path, 'wb') as f:
                    pickle.dump(model_info, f)
                
                print(f"模型已保存至: {save_path}")
                
                # 通知 DataManager 重新加载模型
                self.data_manager.load_models_from_disk()
                
                QMessageBox.information(self, "成功", f"模型 '{name}' 已保存至磁盘，可用于推测页面。")
                
            except Exception as e:
                print(f"保存模型失败: {e}")
                traceback.print_exc()
                QMessageBox.critical(self, "错误", f"保存模型失败: {e}")

    def _clear_plot_layout(self):
        """清除滚动区域布局中的所有旧图形"""
        while self.ui.plot_layout.count():
            child = self.ui.plot_layout.takeAt(0)
            if child.widget():
                # Matplotlib Canvas 需要正确关闭和删除
                if isinstance(child.widget(), FigureCanvas):
                    # 关闭 Figure 以释放资源
                    try:
                        child.widget().figure.clear()
                        plt.close(child.widget().figure)  # 关闭 matplotlib figure
                    except Exception as e:
                        print(f"关闭旧 Figure 时出错: {e}")
                child.widget().deleteLater()  # 安排 Qt 删除控件

    def _get_cluster_colors(self, n_clusters):
        """获取用于聚类的颜色列表"""
        # 自定义颜色列表
        custom_colors = ['#FF0000', '#FFA500', '#FFFF00', '#008000',
                         '#00BFFF', '#00FFFF', '#800080', '#FFC0CB',
                         '#A52A2A', '#FFD700', '#00FF7F', '#7B68EE',
                         '#C0C0C0', '#000000', '#FFF8DC', '#808000']
        colors = [custom_colors[i % len(custom_colors)] for i in range(n_clusters)]
        return colors

    def update_plots(self, signal_data, peak_indices, cluster_labels, peak_features=None, signal_ylabel="幅值"):
        """
        更新滚动绘图区域，显示多个分析图形。
        :param signal_data: 原始信号数据 (一维数组或列表)
        :param peak_indices: 检测到的峰值的索引 (一维数组或列表)
        :param cluster_labels: K-Means (或其他算法) 返回的聚类标签 (与 peak_indices 对应)
        :param peak_features: 包含峰值特征的结构 (例如，NxM NumPy 数组，N是峰值数，M是特征数，如高度、宽度)
        :param signal_ylabel: 原始信号Y轴的标签文本
        """
        print("准备更新滚动绘图区域...")
        self._clear_plot_layout()  # 清除旧图形

        if cluster_labels is None or len(peak_indices) != len(cluster_labels):
            print("错误：标签数据无效或与峰值数量不匹配。")
            self.ui._display_plot_error("无效的聚类结果或数据不匹配。")
            return

        try:
            unique_labels = sorted(list(np.unique(cluster_labels)))
            n_clusters = len(unique_labels)
            if n_clusters == 0:
                self.ui._display_plot_error("没有找到有效的聚类。")
                return

            print(f"检测到 {n_clusters} 个类别: {unique_labels}")

            # --- 1. 绘制主信号图 ---
            self._plot_main_signal(signal_data, peak_indices, cluster_labels, signal_ylabel)

            # --- 2. 绘制每个类别的平均波形图 ---
            waveform_window = 100  # 峰值左右各 50 个点
            self._plot_average_waveforms(signal_data, peak_indices, cluster_labels, unique_labels, waveform_window)

            # --- 3. 绘制特征箱式图 (如果提供了特征数据) ---
            if peak_features is not None and peak_features.shape[0] == len(peak_indices):
                num_features = peak_features.shape[1]
                feature_names = ['heights', 'widths', 'prominences']
                self._plot_feature_boxplots(peak_features, cluster_labels, unique_labels, feature_names)
            else:
                print("跳过特征箱式图：未提供特征数据。")

            # 可能需要强制刷新布局
            self.ui.scroll_content_widget.adjustSize()  # 调整内容控件大小以适应其内容
            print("所有绘图已添加到滚动区域。")

        except Exception as e:
            print(f"更新多个绘图时出错: {e}")
            traceback.print_exc()
            self._clear_plot_layout()  # 出错时也清除一下
            self.ui._display_plot_error(f"绘图时发生错误:\n{e}")

    def _plot_main_signal(self, signal_data, peak_indices, cluster_labels, ylabel):
        """绘制主信号和聚类峰值"""
        print("绘制主信号图...")
        figure, canvas = self.ui._create_figure_canvas(fixed_height=700)  # 主图高一点

        # 降采样处理
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
            x_val = start  # 以起始下标为 x 轴
            y_min = np.nanmin(segment)
            y_max = np.nanmax(segment)

            idx = valid_points * 2
            x_display[idx] = x_val
            y_display[idx] = y_min
            x_display[idx + 1] = x_val
            y_display[idx + 1] = y_max
            valid_points += 1

        x_display = x_display[:valid_points * 2]
        y_display = y_display[:valid_points * 2]

        # 创建绘图对象
        ax = figure.add_subplot(111)

        # 绘制降采样后的信号
        ax.plot(x_display, y_display, label='原始信号', color='blue', zorder=1)

        # 绘制聚类点
        unique_labels = sorted(list(np.unique(cluster_labels)))
        colors = self._get_cluster_colors(len(unique_labels))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        plotted_labels = set()

        for i in range(len(peak_indices)):
            peak_idx = peak_indices[i]
            label = cluster_labels[i]
            color = colors[label_map[label]]
            label_text = f'类别 {label}'
            if label not in plotted_labels:
                ax.plot(peak_idx, signal_data[peak_idx], 'o', markersize=6, color=color, label=label_text, zorder=2)
                plotted_labels.add(label)
            else:
                ax.plot(peak_idx, signal_data[peak_idx], 'o', markersize=6, color=color, zorder=2)

        # 设置标题、标签等
        ax.set_title(f'{self.ui.function_selector.currentText()} 聚类结果 - 主信号图',
                     fontproperties=get_chinese_font(), fontsize=18)
        ax.set_xlabel("时间点", fontproperties=get_chinese_font())
        ax.set_ylabel(ylabel if ylabel else "幅值", fontproperties=get_chinese_font())
        ax.legend(prop=get_chinese_font(), fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)

        figure.tight_layout()
        canvas.draw()
        print("主图已绘制。")
        self.ui._add_plot_to_layout(canvas)

    def _plot_average_waveforms(self, signal_data, peak_indices, cluster_labels, unique_labels, window_size):
        """为每个类别绘制平均峰值波形"""
        print("绘制平均波形图...")
        colors = self._get_cluster_colors(len(unique_labels))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        signal_len = len(signal_data)

        for label in unique_labels:
            figure, canvas = self.ui._create_figure_canvas(fixed_height=500)
            ax = figure.add_subplot(111)

            cluster_peak_indices = peak_indices[cluster_labels == label]
            waveforms = []
            for peak_idx in cluster_peak_indices:
                start = max(0, peak_idx - window_size // 2)
                end = min(signal_len, peak_idx + window_size // 2)
                waveform = signal_data[start:end]
                if len(waveform) > 0:  # 确保不是空片段
                    waveforms.append(waveform)

            if not waveforms:
                print(f"类别 {label} 没有有效的波形数据可用于绘制平均图。")
                plt.close(figure)  # 关闭这个空的 figure
                continue  # 跳过这个类别的绘图

            # 处理不等长波形: 找到最大长度，用 NaN 填充较短的
            max_len = max(len(wf) for wf in waveforms)
            padded_waveforms = []
            for wf in waveforms:
                padding = max_len - len(wf)
                # 使用 np.nan 填充，这样 nanmean 会忽略它们
                padded_wf = np.pad(wf.astype(float), (0, padding), 'constant', constant_values=np.nan)
                padded_waveforms.append(padded_wf)

            if not padded_waveforms:
                print(f"类别 {label} 填充后没有有效的波形数据。")
                plt.close(figure)  # 关闭这个空的 figure
                continue

            # 计算平均波形，忽略 NaN
            avg_waveform = np.nanmean(np.array(padded_waveforms), axis=0)
            std_waveform = np.nanstd(np.array(padded_waveforms), axis=0)  # 计算标准差用于绘制置信区间

            time_axis = np.arange(len(avg_waveform)) - window_size // 2  # 创建相对时间轴

            color_idx = label_map[label]
            color = colors[color_idx]

            # 绘制平均波形
            ax.plot(time_axis, avg_waveform, color=color, linewidth=2, label=f'平均波形 (N={len(waveforms)})')
            # 绘制置信区间
            ax.fill_between(time_axis, avg_waveform - std_waveform, avg_waveform + std_waveform,
                            color=color, alpha=0.2, label='±1 标准差')

            ax.set_title(f'类别 {label} - 平均峰值波形', fontproperties=get_chinese_font(), fontsize=18)
            ax.set_xlabel("时间", fontproperties=get_chinese_font(), fontsize=14)
            ax.set_ylabel("电流", fontproperties=get_chinese_font(), fontsize=14)
            ax.legend(prop=get_chinese_font())
            ax.grid(True, linestyle='--', alpha=0.6)
            figure.tight_layout()
            canvas.draw()
            self.ui._add_plot_to_layout(canvas)

    def _plot_feature_boxplots(self, peak_features, cluster_labels, unique_labels, feature_names):
        """为每个特征绘制按类别分组的箱式图"""
        print("绘制特征箱式图...")
        num_features = peak_features.shape[1]
        colors = self._get_cluster_colors(len(unique_labels))  # 获取颜色

        # 为每个特征创建一个图
        for feat_idx in range(num_features):
            figure, canvas = self.ui._create_figure_canvas(figsize=(max(6, len(unique_labels) * 0.8), 4),
                                                           fixed_height=500)  # 根据类别数量调整宽度
            ax = figure.add_subplot(111)

            data_to_plot = []
            plot_labels = []

            box_colors = []  # 每个箱子的颜色
            label_map = {label: i for i, label in enumerate(unique_labels)}

            for label in unique_labels:
                label_mask = (cluster_labels == label)
                feature_data = peak_features[label_mask, feat_idx]
                if len(feature_data) > 0:  # 确保有数据
                    data_to_plot.append(feature_data)
                    plot_labels.append(f'类别 {label}')
                    color_idx = label_map[label]
                    box_colors.append(colors[color_idx])
                else:
                    # 如果某个类别没有数据，跳过
                    pass

            if not data_to_plot:
                print(f"特征 '{feature_names[feat_idx]}' 没有足够的数据进行箱式图绘制。")
                plt.close(figure)  # 关闭空 figure
                continue  # 跳到下一个特征

            # 创建箱式图
            bp = ax.boxplot(data_to_plot, patch_artist=True,  # 允许填充颜色
                            showfliers=False, )  # 不显示异常值点，让图更清晰

            # 为每个箱子设置颜色
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)  # 设置透明度

            # 设置中位线颜色为黑色，更清晰
            for median in bp['medians']:
                median.set_color('black')

            ax.set_title(f'特征 "{feature_names[feat_idx]}" 按类别分布', fontproperties=get_chinese_font(), fontsize=18)
            ax.set_ylabel(feature_names[feat_idx], fontproperties=get_chinese_font(), fontsize=14)

            # 设置 X 轴标签
            ax.set_xticklabels(plot_labels, fontproperties=get_chinese_font())

            # 如果类别标签太长或太多，旋转它们
            if len(plot_labels) > 5:
                ax.tick_params(axis='x', rotation=45)
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)  # 只显示水平网格线
            figure.tight_layout()  # 调整布局以适应旋转的标签
            canvas.draw()
            self.ui._add_plot_to_layout(canvas)

    def save_data(self):
        """保存当前滚动区域中显示的所有图形到一个指定的文件夹"""
        print("开始保存聚类图形...")

        # 1. 检查是否有图形可保存
        if self.ui.plot_layout.count() == 0:
            QMessageBox.warning(self, "无法保存", "当前没有聚类图形可供保存。")
            print("没有图形需要保存。")
            return

        # 2. 获取当前选中的数据源名称（用于生成文件夹名）
        current_item = self.ui.submission_list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "无法保存", "请先在左侧列表中选择一个数据源。")
            print("未选择数据源。")
            return

        selected_data_name = current_item.text()
        # 清理数据源名称，使其适合作为文件夹名的一部分
        base_foldername = re.sub(r'[\\/*?:"<>|]', '_', selected_data_name)  # 替换非法字符

        # 3. 选择保存目录
        default_dir = os.path.expanduser("~")  # 默认用户主目录
        save_dir = QFileDialog.getExistingDirectory(self, "选择保存图形的文件夹", default_dir)

        if not save_dir:
            print("用户取消保存。")
            return

        # 4. 创建主文件夹和子文件夹
        timestamp = time.strftime("%Y%m%d_%H%M%S")  # 添加时间戳避免覆盖
        main_folder_name = f"{base_foldername}_clustering_{timestamp}"
        main_folder_path = os.path.join(save_dir, main_folder_name)
        plots_dir = os.path.join(main_folder_path, "clustering_plots")

        try:
            os.makedirs(plots_dir, exist_ok=True)  # 创建 clustering_plots 文件夹
            print(f"图形将保存到: {plots_dir}")
        except OSError as e:
            QMessageBox.critical(self, "创建文件夹失败", f"无法创建保存目录:\n{plots_dir}\n错误: {e}")
            print(f"创建目录失败: {e}")
            return

        # 5. 迭代保存图形
        num_plots = self.ui.plot_layout.count()
        saved_count = 0
        error_count = 0

        # 添加进度对话框
        progress = QProgressDialog(f"正在保存 {num_plots} 个图形...", "取消", 0, num_plots, self)
        progress.setWindowTitle("保存进度")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(500)  # 0.5秒后显示
        progress.setValue(0)
        QApplication.processEvents()  # 确保对话框显示

        plot_index = 0  # 用于文件命名计数
        for i in range(num_plots):
            if progress.wasCanceled():
                print("保存被用户取消。")
                break

            widget = self.ui.plot_layout.itemAt(i).widget()

            if isinstance(widget, FigureCanvas):
                plot_index += 1  # 只对 FigureCanvas 计数
                progress.setLabelText(f"正在保存图形 {plot_index}/{num_plots}...")
                progress.setValue(plot_index)
                QApplication.processEvents()

                figure = widget.figure
                # 尝试从图形标题生成更具体的文件名
                filename_base = f"plot_{plot_index:02d}"  # 默认文件名 plot_01, plot_02, ...
                try:
                    if figure.axes:  # 检查是否有坐标轴
                        ax = figure.axes[0]  # 通常取第一个坐标轴
                        title = ax.get_title()
                        if title:
                            # 尝试提取关键信息
                            if "主信号图" in title or "Main Signal" in title:
                                filename_base += "_main_signal"
                            elif "平均峰值波形" in title or "Average Waveform" in title:
                                match = re.search(r'类别\s*(\S+)', title)  # 查找 "类别 X"
                                if match:
                                    label = match.group(1).replace(':', '')  # 提取标签并清理
                                    filename_base += f"_avg_waveform_cluster_{label}"
                                else:
                                    filename_base += "_avg_waveform"  # 备用
                            elif "特征" in title or "Feature" in title:
                                match = re.search(r'特征\s*"([^"]+)"', title)  # 查找 "特征 \"X\""
                                if match:
                                    feature_name = match.group(1)
                                    sanitized_feature = re.sub(r'\W+', '_', feature_name)  # 清理特征名
                                    filename_base += f"_feature_{sanitized_feature}"
                                    if "箱式图" in title or "Boxplot" in title:  # 检查是否是箱式图
                                        filename_base += "_boxplot"
                                else:
                                    filename_base += "_feature_plot"  # 备用
                            elif "散点图" in title or "Scatter Plot" in title:
                                filename_base += "_scatter_plot"
                            # 进一步清理可能包含的非法字符
                            filename_base = re.sub(r'[\\/*?:"<>|]', '_', filename_base)

                except Exception as e_title:
                    print(f"从标题获取文件名信息时出错: {e_title}")
                    # 出错时继续使用默认文件名

                # 最终文件名和路径
                save_filename = f"{filename_base}.png"
                filepath = os.path.join(plots_dir, save_filename)

                # 保存图形
                try:
                    figure.savefig(filepath, dpi=200, bbox_inches='tight')  # 保存为 PNG
                    saved_count += 1
                    print(f"已保存: {save_filename}")
                except Exception as e_save:
                    error_count += 1
                    print(f"保存图形 '{save_filename}' 失败: {e_save}")

        progress.setValue(num_plots)  # 完成进度条
        QApplication.processEvents()

        # 6. 显示最终结果
        if not progress.wasCanceled():
            if error_count == 0:
                QMessageBox.information(self, "保存完成",
                                        f"成功保存 {saved_count} 个图形到:\n{plots_dir}")
            else:
                QMessageBox.warning(self, "保存部分完成",
                                    f"成功保存 {saved_count} 个图形，但有 {error_count} 个保存失败。\n保存在:\n{plots_dir}")
        else:
            QMessageBox.information(self, "保存已取消",
                                    f"保存过程被取消。\n已保存 {saved_count} 个图形到:\n{plots_dir}")

        print(f"--- 图形保存结束 (成功: {saved_count}, 失败: {error_count}) ---")
