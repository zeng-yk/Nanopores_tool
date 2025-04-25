# clustering.py
import threading

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy, QHBoxLayout, QPushButton, QStackedWidget, \
    QComboBox, QFrame, QSplitter, QListWidget, QApplication, QScrollArea
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib import font_manager

from parameter_widgets import (BaseParameterWidget, KMeansParameterWidget,
                               DBSCANParameterWidget, PlaceholderParameterWidget)

import matplotlib

matplotlib.use('Qt5Agg')  # 指定使用 Qt5 后端
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt  # 仍然需要 plt 来获取颜色映射等
import platform


def get_chinese_font():
    system = platform.system()
    if system == "Windows":
        # Windows 常见中文字体
        font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
    elif system == "Darwin":
        # macOS 使用 STHeiti 或 PingFang
        font_path = "/System/Library/Fonts/STHeiti Light.ttc"

    return font_manager.FontProperties(fname=font_path)


class ClusteringPage(QWidget):

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.results = None
        self.data_manager = data_manager

        self.param_widgets = {}  # 用于存储参数控件实例的字典

        self.signal_for_plot = None
        self.peaks_for_plot = None
        self.labels_for_plot = None
        self.features_for_plot = None

        self.setup_ui()
        self.data_manager.submissions_changed_signal.connect(self.update_submission_list)
        # 初始化时显式更新一次列表
        self.update_submission_list()  # <-- 添加这一行

    def setup_ui(self):
        main_layout = QHBoxLayout(self)  # 主窗口水平布局

        # --- 左侧面板 (垂直 QSplitter 分隔) ---
        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.setObjectName("left_splitter")
        # 添加简单的分隔条样式，使其可见
        # left_splitter.setStyleSheet("""
        #             QSplitter::handle {
        #                 background-color: #cccccc; /* Light gray handle */
        #             }
        #             QSplitter::handle:vertical {
        #                 height: 4px; /* Handle thickness */
        #             }
        #             QSplitter::handle:pressed {
        #                 background-color: #aaaaaa; /* Darker gray when pressed */
        #             }
        #         """)
        # --- 配置数据列表区 ---
        self.setup_data_list_area(left_splitter)

        # 2. 下拉列表切换区
        self.setup_dropdown_area(left_splitter)  # 封装下拉列表设置

        # 3. 参数设置区 (现在包含 QStackedWidget)
        self.setup_parameter_area(left_splitter)  # 封装参数区设置

        self.button = QPushButton("运行聚类")
        self.button.clicked.connect(self.run_selected_algorithm)
        left_splitter.addWidget(self.button)

        self.save_button = QPushButton("保存结果")
        # self.save_button.clicked.connect(self.run_selected_algorithm)
        left_splitter.addWidget(self.save_button)

        # 设置左侧 Splitter 的初始大小 (大致比例)
        left_splitter.setSizes([150, 80, 300, 10, 10])

        main_layout.addWidget(left_splitter)

        self.setup_plot_area(main_layout)

        # --- 右侧面板 (占位符) ---
        # self.main_plot_area = QWidget()
        # self.main_plot_area.setObjectName("main_plot_area")
        # main_plot_layout = QVBoxLayout(self.main_plot_area)
        # main_plot_label = QLabel("主图")
        # main_plot_label.setAlignment(Qt.AlignCenter)
        # main_plot_layout.addWidget(main_plot_label)
        # # 基本样式
        # self.main_plot_area.setStyleSheet('''
        #             QWidget#main_plot_area {
        #                 background-color: #e0e0e0;
        #                 border: 1px solid #999999;
        #             }
        #              QLabel { font-size: 20px; color: #666; }
        #         ''')

        # --- 组合主布局 ---
        # main_layout.addWidget(left_splitter)
        # main_layout.addWidget(self.main_plot_area)

        # 设置主布局拉伸因子 (右侧更宽)
        main_layout.setStretchFactor(left_splitter, 2)
        main_layout.setStretchFactor(self.scroll_area, 8)

        # --- 连接信号 ---
        # 连接下拉列表的信号到一个简单的处理函数
        self.function_selector.currentIndexChanged.connect(self.on_function_changed)

    def setup_data_list_area(self, parent_splitter: QSplitter):
        """配置数据加载区域的 UI 元素 (仅显示列表)。"""
        self.data_load_area = QWidget()
        self.data_load_area.setObjectName("data_load_area")
        data_load_layout = QVBoxLayout(self.data_load_area)
        data_load_layout.setContentsMargins(5, 5, 5, 5)
        data_load_layout.setSpacing(5)

        # 1. 标题标签
        data_load_label = QLabel("已提交的识别数据")
        data_load_label.setAlignment(Qt.AlignCenter)
        # data_load_label.setFont(QFont("微软雅黑", 12, QFont.Bold))
        data_load_layout.addWidget(data_load_label)

        # 2. 列表控件，用于显示 submission 名称
        self.submission_list_widget = QListWidget()
        self.submission_list_widget.setObjectName("submissionList")

        # 使列表项不可选择或交互
        # self.submission_list_widget.setSelectionMode(QListWidget.NoSelection)
        # self.submission_list_widget.setFocusPolicy(Qt.NoFocus)

        data_load_layout.addWidget(self.submission_list_widget)

        # 更新样式表，移除按钮相关样式
        self.data_load_area.setStyleSheet('''
            QWidget#data_load_area {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 4px;
                /* min-height: 150px; */ /* 可根据内容调整 */
            }
            QLabel {
                font-size: 14px;
                color: #333;
                padding-bottom: 5px;
                font-weight: bold;
            }
            QListWidget#submissionList {
                background-color: white;
                border: 1px solid #bbbbbb;
                font-size: 13px;
            }
            /* 移除了选中项样式，如果设置了 NoSelection */
            /* 如果仍允许选择，可以保留选中样式 */
            QListWidget#submissionList::item:selected {
                 background-color: #e0e0e0; /* 可以改为灰色或移除 */
                 color: black;
            }
        ''')

        parent_splitter.addWidget(self.data_load_area)

    def update_submission_list(self):
        """
        刷新 QListWidget 列表，使其显示 DataManager 中当前所有 submission 的 'name'。
        """
        print("UI: 开始更新 submission 名称列表...")

        # 1. 从 DataManager 获取最新的 'name' 列表
        #    这里调用了我们之前定义的 get_submission_names 方法
        submission_names = self.data_manager.get_submission_names()

        # 2. 清空 QListWidget 中的所有旧项目
        self.submission_list_widget.clear()

        # 3. 检查获取到的名称列表是否为空
        if submission_names:
            # 如果列表不为空，则将列表中的所有名称作为新项目添加到 QListWidget 中
            # addItems 方法可以直接接受一个字符串列表
            self.submission_list_widget.addItems(submission_names)
            print(f"UI: QListWidget 已填充以下名称: {submission_names}")
            # QApplication.processEvents()  # <--- 临时添加以强制处理事件
        else:
            # 如果列表为空，打印提示信息
            print("UI: DataManager 中没有 submission 名称可供显示。")

        print("UI: 列表更新完成。")

    def setup_dropdown_area(self, parent_splitter: QSplitter):
        # 2. 下拉列表切换区
        self.dropdown_switch_area = QWidget()
        self.dropdown_switch_area.setObjectName("dropdown_switch_area")
        # 使用 Frame 增加边框感
        dropdown_frame = QFrame(self.dropdown_switch_area)
        dropdown_frame.setFrameShape(QFrame.StyledPanel)

        dropdown_area_layout = QVBoxLayout(dropdown_frame)
        dropdown_area_layout.setContentsMargins(5, 5, 5, 5)

        # ** 重要：在这里定义下拉选项和映射 **
        self.algorithm_map = {
            "K-Means": KMeansParameterWidget,
            "DBSCAN": DBSCANParameterWidget,
            "功能 3": PlaceholderParameterWidget
        }
        self.function_selector = QComboBox()
        self.function_selector.addItems(self.algorithm_map.keys())  # 从映射的键添加项
        self.function_selector.setObjectName("function_selector")

        dropdown_area_layout.addWidget(QLabel("聚类算法:"))  # 标签放在组合框上方
        dropdown_area_layout.addWidget(self.function_selector)

        container_layout = QVBoxLayout(self.dropdown_switch_area)
        container_layout.setContentsMargins(5, 5, 5, 5)  # 外边距
        container_layout.addWidget(dropdown_frame)

        # 基本样式
        self.dropdown_switch_area.setStyleSheet('''
                            QWidget#dropdown_switch_area { /* Container widget styling */
                                 /* background-color: lightblue; */ /* Optional debug color */
                                 padding: 0px; /* No padding on container */
                            }
                            QFrame { /* Frame styling */
                                 background-color: #e8e8e8;
                                 border: 1px solid #bbbbbb;
                                 border-radius: 3px;
                            }
                            QLabel { margin-bottom: 2px; font-size: 12px; color: #333;}
                            QComboBox#function_selector { min-height: 22px; }
                        ''')
        # 固定下拉区域的高度，使其紧凑
        self.dropdown_switch_area.setFixedHeight(self.dropdown_switch_area.sizeHint().height() + 5)  # Add padding
        parent_splitter.addWidget(self.dropdown_switch_area)

    def setup_parameter_area(self, parent_splitter: QSplitter):
        # 3. 参数设置区 (占位符)
        self.parameter_settings_area = QWidget()
        self.parameter_settings_area.setObjectName("parameter_settings_area")
        parameter_settings_layout = QVBoxLayout(self.parameter_settings_area)
        parameter_settings_layout.setContentsMargins(5, 5, 5, 5)
        parameter_settings_layout.setSpacing(0)

        self.parameter_stack = QStackedWidget()  # 创建 Stacked Widget
        parameter_settings_layout.addWidget(self.parameter_stack)

        # 实例化每个参数控件并添加到 Stacked Widget 和字典中
        for name, WidgetClass in self.algorithm_map.items():
            widget_instance = WidgetClass()
            self.parameter_stack.addWidget(widget_instance)
            self.param_widgets[name] = widget_instance  # 存储实例引用
            print(f"添加参数页面: {name} -> {WidgetClass.__name__}")

        # 基本样式
        self.parameter_settings_area.setStyleSheet('''
                    QWidget#parameter_settings_area {
                        background-color: #f0f0f0;
                        border: 1px solid #aaaaaa;
                        min-height: 200px; /* 最小高度 */
                    }
                    QLabel { font-size: 14px; color: #555; }
                ''')
        parent_splitter.addWidget(self.parameter_settings_area)

    def setup_plot_area(self, parent_layout: QHBoxLayout):
        """设置主绘图区域，嵌入 Matplotlib 画布"""
        # 1. 创建 QScrollArea
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("main_plot_scroll_area")
        self.scroll_area.setWidgetResizable(True)  # 关键：允许内部控件调整大小
        self.scroll_area.setStyleSheet("""
                    QScrollArea#main_plot_scroll_area {
                        border: 1px solid #999999;
                    }
                """)

        # 2. 创建 QScrollArea 内部的容器 QWidget
        #    所有图形都将添加到这个容器的布局中
        self.scroll_content_widget = QWidget()
        self.scroll_content_widget.setObjectName("scroll_content_widget")
        # self.scroll_content_widget.setStyleSheet("background-color: #f0f0f0;") # 可选：给内容区域一个背景色

        # 3. 为内部容器创建一个垂直布局
        self.plot_layout = QVBoxLayout(self.scroll_content_widget)
        self.plot_layout.setContentsMargins(5, 5, 5, 5)  # 设置边距
        self.plot_layout.setSpacing(10)  # 设置图形之间的间距

        # 4. 将内部容器设置为 QScrollArea 的控件
        self.scroll_area.setWidget(self.scroll_content_widget)

        # 5. 将 QScrollArea 添加到父布局
        parent_layout.addWidget(self.scroll_area)  # 第二个参数表示拉伸因子

    # --- 新的绘图函数，绘制多个图形 ---
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
            self._display_plot_error("无效的聚类结果或数据不匹配。")
            return

        try:
            unique_labels = sorted(list(np.unique(cluster_labels)))
            n_clusters = len(unique_labels)
            if n_clusters == 0:
                self._display_plot_error("没有找到有效的聚类。")
                return

            print(f"检测到 {n_clusters} 个类别: {unique_labels}")

            # --- 1. 绘制主信号图 (类似原来的图) ---
            self._plot_main_signal(signal_data, peak_indices, cluster_labels, signal_ylabel)

            # --- 2. 绘制每个类别的平均波形图 ---
            # 需要一个窗口大小来提取峰值周围的波形
            waveform_window = 100  # 示例窗口大小 (峰值左右各 50 个点)，需要根据实际情况调整
            self._plot_average_waveforms(signal_data, peak_indices, cluster_labels, unique_labels, waveform_window)

            # # --- 3. 绘制特征散点图 (如果提供了特征数据) ---
            # if peak_features is not None and peak_features.shape[0] == len(peak_indices) and peak_features.shape[
            #     1] >= 2:
            #     # 假设使用前两个特征进行绘图
            #     feature_x_index = 0
            #     feature_y_index = 1
            #     feature_x_name = "特征 1 (高度)"  # 替换为实际名称
            #     feature_y_name = "特征 2 (宽度)"  # 替换为实际名称
            #     self._plot_feature_scatter(peak_features, cluster_labels, unique_labels,
            #                                feature_x_index, feature_y_index,
            #                                feature_x_name, feature_y_name)
            # else:
            #     print("跳过特征散点图：未提供足够的特征数据。")

            # --- 4. 绘制特征箱式图 (如果提供了特征数据) ---
            if peak_features is not None and peak_features.shape[0] == len(peak_indices):
                # 为每个特征绘制箱式图
                num_features = peak_features.shape[1]
                # feature_names = [f"特征 {i + 1}" for i in range(num_features)]  # 使用通用名称，最好替换为实际名称
                feature_names = ['heights', 'widths', 'prominences']
                self._plot_feature_boxplots(peak_features, cluster_labels, unique_labels, feature_names)
            else:
                print("跳过特征箱式图：未提供特征数据。")

            # 可能需要强制刷新布局，尽管 QScrollArea 通常会自动处理
            self.scroll_content_widget.adjustSize()  # 调整内容控件大小以适应其内容
            print("所有绘图已添加到滚动区域。")

        except Exception as e:
            print(f"更新多个绘图时出错: {e}")
            import traceback
            traceback.print_exc()
            self._clear_plot_layout()  # 出错时也清除一下
            self._display_plot_error(f"绘图时发生错误:\n{e}")

    def _create_figure_canvas(self, figsize=(6, 4), dpi=100, fixed_height=400):
        """辅助函数：创建一个新的 Figure, Canvas 并设置固定高度"""
        figure = Figure(figsize=figsize, dpi=dpi)
        canvas = FigureCanvas(figure)
        if fixed_height:
            # 设置固定高度对于 QScrollArea 内的垂直布局很重要
            canvas.setFixedHeight(fixed_height)
        return figure, canvas

    def _add_plot_to_layout(self, canvas):
        """辅助函数：将 Canvas 添加到滚动区域的布局中"""
        self.plot_layout.addWidget(canvas)

    def _get_cluster_colors(self, n_clusters):
        """获取用于聚类的颜色列表"""
        # 你可以继续使用你的自定义颜色，或者用 colormap
        custom_colors = ['#FF0000', '#FFA500', '#FFFF00', '#008000',
                         '#00BFFF', '#00FFFF', '#800080', '#FFC0CB',
                         '#A52A2A', '#FFD700', '#00FF7F', '#7B68EE',
                         '#C0C0C0', '#000000', '#FFF8DC', '#808000']
        # cmap = plt.cm.get_cmap('tab10', max(1, n_clusters)) # 使用 tab10 colormap
        # colors = [cmap(i) for i in range(n_clusters)]
        colors = [custom_colors[i % len(custom_colors)] for i in range(n_clusters)]
        return colors

    def _display_plot_error(self, message):
        """在绘图区域显示错误信息"""
        figure, canvas = self._create_figure_canvas(fixed_height=200)
        ax = figure.add_subplot(111)
        ax.text(0.5, 0.5, message,
                horizontalalignment='center', verticalalignment='center',
                fontsize=14, fontweight='bold', color='red',
                fontproperties=get_chinese_font(), wrap=True)  # wrap=True 允许文本换行
        ax.axis('off')  # 关闭坐标轴
        figure.tight_layout()
        canvas.draw()
        self._add_plot_to_layout(canvas)

        # --- 绘图辅助函数 ---

    def _plot_main_signal(self, signal_data, peak_indices, cluster_labels, ylabel):
        """绘制主信号和聚类峰值"""
        print("绘制主信号图...")
        figure, canvas = self._create_figure_canvas(fixed_height=600)  # 主图高一点
        # ax = figure.add_subplot(111)
        #
        # # 绘制原始信号
        # ax.plot(signal_data, label='信号', color='blue', zorder=1)  # 降低信号线的 zorder
        #
        # # 准备颜色和标签
        # unique_labels = sorted(list(np.unique(cluster_labels)))
        # colors = self._get_cluster_colors(len(unique_labels))
        # label_map = {label: i for i, label in enumerate(unique_labels)}  # 将原始标签映射到颜色索引
        #
        # plotted_labels = set()
        # for i in range(len(peak_indices)):
        #     peak_idx = peak_indices[i]
        #     label = cluster_labels[i]
        #     color_idx = label_map[label]
        #     color = colors[color_idx]
        #
        #     label_text = f'类别 {label}'
        #     if label not in plotted_labels:
        #         ax.plot(peak_idx, signal_data[peak_idx], 'o', markersize=6,
        #                 color=color, label=label_text, zorder=2)
        #         plotted_labels.add(label)
        #     else:
        #         ax.plot(peak_idx, signal_data[peak_idx], 'o', markersize=6, color=color, zorder=2)
        #
        # ax.set_title(f'{self.function_selector.currentText()} 聚类结果 - 主信号图',
        #              fontproperties=get_chinese_font())
        # ax.set_xlabel("时间点", fontproperties=get_chinese_font())
        # ax.set_ylabel(ylabel if ylabel else "幅值", fontproperties=get_chinese_font())
        # ax.legend(prop=get_chinese_font())
        #
        # ax.grid(True, linestyle='--', alpha=0.6)  # 添加网格线
        #
        # figure.tight_layout()  # 8. 调整布局防止标签重叠
        # canvas.draw()
        # print("主图已绘制。")
        # self._add_plot_to_layout(canvas)

        # 降采样参数
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
        ax.set_title(f'{self.function_selector.currentText()} 聚类结果 - 主信号图', fontproperties=get_chinese_font())
        ax.set_xlabel("时间点", fontproperties=get_chinese_font())
        ax.set_ylabel(ylabel if ylabel else "幅值", fontproperties=get_chinese_font())
        ax.legend(prop=get_chinese_font())
        ax.grid(True, linestyle='--', alpha=0.6)

        figure.tight_layout()
        canvas.draw()
        print("主图已绘制。")
        self._add_plot_to_layout(canvas)

    def _plot_average_waveforms(self, signal_data, peak_indices, cluster_labels, unique_labels, window_size):
        """为每个类别绘制平均峰值波形"""
        print("绘制平均波形图...")
        colors = self._get_cluster_colors(len(unique_labels))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        signal_len = len(signal_data)

        for label in unique_labels:
            figure, canvas = self._create_figure_canvas(fixed_height=300)  # 平均波形图可以矮一点
            ax = figure.add_subplot(111)

            cluster_peak_indices = peak_indices[cluster_labels == label]
            waveforms = []
            for peak_idx in cluster_peak_indices:
                start = max(0, peak_idx - window_size // 2)
                end = min(signal_len, peak_idx + window_size // 2)
                waveform = signal_data[start:end]
                # 如果波形不够长（边界情况），可能需要填充或跳过
                # 为了简单起见，这里假设都能取到足够长度，或者绘制不等长波形
                # 更稳健的方法是确保所有提取的波形长度一致，例如通过填充
                # 这里我们先直接收集，如果长度不一，求平均值时需要处理
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
            # 绘制置信区间 (可选)
            ax.fill_between(time_axis, avg_waveform - std_waveform, avg_waveform + std_waveform,
                            color=color, alpha=0.2, label='±1 标准差')

            ax.set_title(f'类别 {label} - 平均峰值波形', fontproperties=get_chinese_font())
            ax.set_xlabel("时间", fontproperties=get_chinese_font())
            ax.set_ylabel("电流", fontproperties=get_chinese_font())
            ax.legend(prop=get_chinese_font())
            ax.grid(True, linestyle='--', alpha=0.6)
            figure.tight_layout()
            canvas.draw()
            self._add_plot_to_layout(canvas)

    def _plot_feature_scatter(self, peak_features, cluster_labels, unique_labels, x_idx, y_idx, x_name, y_name):
        """绘制峰值特征的散点图"""
        print(f"绘制特征散点图 ({x_name} vs {y_name})...")
        figure, canvas = self._create_figure_canvas(fixed_height=350)
        ax = figure.add_subplot(111)

        colors = self._get_cluster_colors(len(unique_labels))
        label_map = {label: i for i, label in enumerate(unique_labels)}

        for label in unique_labels:
            label_mask = (cluster_labels == label)
            features_subset = peak_features[label_mask]
            if features_subset.shape[0] > 0:  # 确保该类别有数据
                color_idx = label_map[label]
                color = colors[color_idx]
                ax.scatter(features_subset[:, x_idx], features_subset[:, y_idx],
                           color=color, label=f'类别 {label}', alpha=0.7, edgecolors='w', s=50)  # s是点的大小

        ax.set_title('峰值特征散点图', fontproperties=get_chinese_font())
        ax.set_xlabel(x_name, fontproperties=get_chinese_font())
        ax.set_ylabel(y_name, fontproperties=get_chinese_font())
        ax.legend(prop=get_chinese_font())
        ax.grid(True, linestyle='--', alpha=0.6)
        figure.tight_layout()
        canvas.draw()
        self._add_plot_to_layout(canvas)

    def _plot_feature_boxplots(self, peak_features, cluster_labels, unique_labels, feature_names):
        """为每个特征绘制按类别分组的箱式图"""
        print("绘制特征箱式图...")
        num_features = peak_features.shape[1]
        colors = self._get_cluster_colors(len(unique_labels))  # 获取颜色

        # 为每个特征创建一个图
        for feat_idx in range(num_features):
            figure, canvas = self._create_figure_canvas(figsize=(max(6, len(unique_labels) * 0.8), 4),
                                                        fixed_height=300)  # 根据类别数量调整宽度
            ax = figure.add_subplot(111)

            data_to_plot = []
            plot_labels = []
            # plot_labels.append(f'Cluster {label}')

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
                    # 如果某个类别没有数据，可以添加一个空列表或跳过
                    # 添加空列表会导致 matplotlib 报错或画出空箱，最好是跳过并在标签中反映
                    # 或者在这里仅收集有数据的类别
                    pass

            if not data_to_plot:
                print(f"特征 '{feature_names[feat_idx]}' 没有足够的数据进行箱式图绘制。")
                plt.close(figure)  # 关闭空 figure
                continue  # 跳到下一个特征

            # 创建箱式图
            bp = ax.boxplot(data_to_plot, patch_artist=True,  # patch_artist=True 允许填充颜色
                            showfliers=False,)  # showfliers=False 不显示异常值点，让图更清晰

            # 为每个箱子设置颜色
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)  # 设置透明度

            # 设置中位线颜色为黑色，更清晰
            for median in bp['medians']:
                median.set_color('black')

            ax.set_title(f'特征 "{feature_names[feat_idx]}" 按类别分布', fontproperties=get_chinese_font())
            ax.set_ylabel(feature_names[feat_idx], fontproperties=get_chinese_font())
            # ax.set_xlabel("聚类类别", fontproperties=get_chinese_font())
            # 如果类别标签太长或太多，旋转它们
            if len(plot_labels) > 5:
                ax.tick_params(axis='x', rotation=45)
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)  # 只显示水平网格线
            figure.tight_layout()  # 调整布局以适应旋转的标签
            canvas.draw()
            self._add_plot_to_layout(canvas)

    # --- 清除滚动区域内容的辅助函数 ---
    def _clear_plot_layout(self):
        """清除滚动区域布局中的所有旧图形"""
        while self.plot_layout.count():
            child = self.plot_layout.takeAt(0)
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

    def on_function_changed(self, index):
        """下拉列表选项改变时的处理函数 (简单示例)"""
        selected_text = self.function_selector.itemText(index)
        print(f"下拉列表切换到: {selected_text} (索引: {index})")
        # 更新参数设置区的标签，以显示切换效果
        if selected_text in self.param_widgets:
            widget_to_show = self.param_widgets[selected_text]
            self.parameter_stack.setCurrentWidget(widget_to_show)
            print(f"参数区显示: {widget_to_show.__class__.__name__}")
        else:
            print(f"错误: 在 param_widgets 中未找到 '{selected_text}'")
            # 可以考虑切换到一个默认的空页面或显示错误信息
            # self.parameter_stack.setCurrentIndex(0) # 假设第一个是占位符

    def get_current_parameters(self) -> dict | None:
        """获取当前显示的参数控件的参数"""
        current_param_widget = self.parameter_stack.currentWidget()
        if isinstance(current_param_widget, BaseParameterWidget):
            try:
                return current_param_widget.get_parameters()
            except NotImplementedError:
                print(f"错误: {current_param_widget.__class__.__name__} 未实现 get_parameters 方法")
                return None
        return None  # 如果当前页面不是参数控件

    def run_selected_algorithm(self):
        """获取参数，运行算法，并更新多个绘图"""
        selected_algorithm_name = self.function_selector.currentText()
        if not selected_algorithm_name:
            print("错误：没有选中任何算法。")
            # QMessageBox.warning(self, "未选择算法", "请先选择一个聚类算法。")
            return
        print(f"选定算法: {selected_algorithm_name}")

        # 2. 获取当前算法的参数
        parameters = self.get_current_parameters()  # 确保 get_current_parameters 能正确工作
        if parameters is None:
            print("错误：无法获取当前算法的参数。")
            # QMessageBox.warning(self, "参数错误", "无法获取算法参数。")
            return
        print(f"获取参数: {parameters}")

        from PyQt5.QtWidgets import QMessageBox
        current_item = self.submission_list_widget.currentItem()
        if not current_item:
            print("错误：请在'已提交的识别数据'中选择要进行聚类的数据。")
            QMessageBox.warning(self, "缺少数据", "请在'已提交的识别数据'中选择要进行聚类的数据项。")
            return

        selected_data_name = current_item.text()
        print(f"目标数据: {selected_data_name}")

        # 调用 DataManager 的方法来获取与该名称对应的数据
        # path_to_process, peaks_to_process, width, half_width, prominences,height = self.data_manager.get_data_by_name(
        #     selected_data_name)
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
        self.features_for_plot = None  # 重置特征

        filtered_features_list = []  # 存储通过过滤的特征 [height, width, prominence]
        for i, peak_idx in enumerate(peaks_to_process):
            filtered_features_list.append([heights[i], widths[i], prominences[i]])

        features_array = np.array(filtered_features_list)
        self.features_for_plot = features_array

        try:
            from algorithms import Algorithms  # 确保导入

            if selected_algorithm_name == "K-Means":
                print("调用 KMeans 算法...")
                # !!! 关键：假设 Algorithms.run_kmeans 现在返回 (signal, peaks, labels, features) !!!
                # 你需要修改你的 Algorithms.run_kmeans 函数来实现这一点
                # features 应该是一个 NumPy 数组，形状为 (n_peaks, n_features)
                result_tuple = Algorithms.run_kmeans(path_to_process, peaks_to_process, parameters)
                if result_tuple and len(result_tuple) == 2:
                    self.signal_for_plot, self.labels_for_plot = result_tuple
                    self.peaks_for_plot = peaks_to_process

                else:
                    print("警告：KMeans 算法未按预期返回信号、标签。可能无法绘制所有图形。")
                    self.labels_for_plot = None  # 标记为无效结果

                    # if result_tuple and len(result_tuple) == 3:  # 兼容旧版
                    #     self.signal_for_plot, self.peaks_for_plot, self.labels_for_plot = result_tuple
                    #     self.features_for_plot = None  # 没有特征数据
                    # else:  # 其他意外情况
                    # self.labels_for_plot = None  # 标记为无效结果


            elif selected_algorithm_name == "DBSCAN":
                print("调用 DBSCAN 算法...")
                # 同样，假设 run_dbscan 也返回 (signal, peaks, labels, features)
                # result_tuple = Algorithms.run_dbscan(path_to_process, data_to_process, parameters)
                # ... (类似 KMeans 的处理逻辑) ...
                print("DBSCAN 的绘图更新逻辑尚未完全实现（需要返回特征）。")
                QMessageBox.information(self, "提示", "DBSCAN 结果绘图（带特征）暂未完全实现。")
                self.labels_for_plot = None  # 暂不处理绘图


            elif selected_algorithm_name == "功能 3":
                print("功能 3 不需要执行聚类算法。")
                self._clear_plot_layout()  # 清除绘图区
                self._display_plot_error("功能 3 无绘图结果。")  # 显示提示信息
                self.labels_for_plot = None  # 无有效标签
                return  # 直接返回


            else:
                QMessageBox.warning(self, "未实现", f"算法 '{selected_algorithm_name}' 的执行逻辑尚未实现。")
                self.labels_for_plot = None
                return

            # --- 更新绘图区 ---
            if self.labels_for_plot is not None and self.signal_for_plot is not None and self.peaks_for_plot is not None:
                print(f"算法执行成功！准备更新 {len(self.labels_for_plot)} 个峰值的绘图...")
                # 注意：现在调用 update_plots，并传入特征数据
                self.update_plots(self.signal_for_plot,
                                  self.peaks_for_plot,
                                  self.labels_for_plot,
                                  self.features_for_plot,  # 传入特征
                                  "电流/NA")  # 你的 Y 轴标签
            elif self.labels_for_plot is None and selected_algorithm_name != "功能 3":  # 如果是算法执行了但没结果
                print("算法执行完成，但没有返回有效的聚类结果或绘图所需数据。")
                self._clear_plot_layout()
                self._display_plot_error("算法未返回有效结果\n或缺少绘图所需数据。")


        except ImportError:
            print("错误：未能导入 algorithms 模块或其中包含的算法库。")
            QMessageBox.critical(self, "导入错误", "运行算法所需的库未能导入，请检查安装。")
            self._clear_plot_layout()
            self._display_plot_error("导入错误，无法运行算法。")
        except Exception as e:
            print(f"运行算法 '{selected_algorithm_name}' 时出错: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "算法错误", f"执行 '{selected_algorithm_name}' 时发生错误:\n{e}")
            self._clear_plot_layout()
            self._display_plot_error(f"算法执行错误:\n{e}")

        print("--- 算法运行与绘图更新结束 ---")
