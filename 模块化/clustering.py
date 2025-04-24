# clustering.py
import threading

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy, QHBoxLayout, QPushButton, QStackedWidget, \
    QComboBox, QFrame, QSplitter, QListWidget, QApplication
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib import font_manager

from parameter_widgets import (BaseParameterWidget, KMeansParameterWidget,
                               DBSCANParameterWidget, PlaceholderParameterWidget)

import matplotlib
matplotlib.use('Qt5Agg') # 指定使用 Qt5 后端
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt # 仍然需要 plt 来获取颜色映射等
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

        # 设置左侧 Splitter 的初始大小 (大致比例)
        left_splitter.setSizes([150, 80, 300, 10])

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
        main_layout.setStretchFactor(left_splitter, 1)
        main_layout.setStretchFactor(self.main_plot_area, 5)

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
        print(f"Update thread: {threading.get_ident()}")
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
        self.main_plot_area = QWidget()
        self.main_plot_area.setObjectName("main_plot_area")
        plot_layout = QVBoxLayout(self.main_plot_area)  # 绘图区的垂直布局

        # 1. 创建 Matplotlib Figure 和 Canvas
        #    我们不使用 plt.figure()，而是手动创建 Figure 对象
        self.figure = Figure(figsize=(5, 4), dpi=100)  # 创建 Figure
        self.canvas = FigureCanvas(self.figure)  # 创建 Canvas，并将 Figure 传入
        plot_layout.addWidget(self.canvas)  # 将 Canvas 添加到布局

        # 2. 添加 Matplotlib 导航工具栏
        # self.toolbar = NavigationToolbar(self.canvas, self)  # 创建工具栏
        # plot_layout.addWidget(self.toolbar)  # 将工具栏添加到布局

        # 设置绘图区样式 (主要是容器的)
        self.main_plot_area.setStyleSheet('''
             QWidget#main_plot_area {
                 /* background-color: #e0e0e0; */ /* 背景现在由画布决定 */
                 border: 1px solid #999999;
             }
         ''')
        parent_layout.addWidget(self.main_plot_area)  # 将包含画布的容器添加到主布局

    # --- 修改 update_plot 方法 ---
    def update_plot(self, first_n_points, valid_peaks, labels, ylabel="Y Axis"):
        """
        更新嵌入的 Matplotlib 画布以显示聚类结果。
        :param first_n_points: 原始信号数据 (一维数组或列表)
        :param valid_peaks: 检测到的峰值的索引 (一维数组或列表)
        :param labels: K-Means 返回的聚类标签 (与 valid_peaks 对应)
        :param ylabel: Y轴的标签文本
        """
        print("准备更新嵌入的 Matplotlib 绘图区域...")
        if labels is None or len(valid_peaks) != len(labels):
            print("错误：标签数据无效或与峰值数量不匹配。")
            # 可以清除画布或显示错误信息
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5,
                    '无效的聚类结果',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14,
                    fontweight='bold',
                    color='red',
                    fontproperties=get_chinese_font()
                    )
            self.canvas.draw()
            return

        try:
            # 1. 清除旧的图形
            self.figure.clear()

            # 2. 添加一个新的子图 (Axes)
            ax = self.figure.add_subplot(111)

            # 3. 绘制原始信号
            ax.plot(first_n_points, label='信号', color='blue')  # 给原始信号一个固定颜色

            # 4. 确定类别数量和颜色映射
            # unique_labels = np.unique(labels)  # 获取所有唯一的类别标签
            # n_clusters = len(unique_labels)
            # 使用 Matplotlib 的颜色映射来自动获取足够多的颜色
            # 'tab10', 'tab20', 'viridis', 'plasma'
            # get_cmap 需要类别数，确保至少为 1
            # cmap = plt.cm.get_cmap('tab20', max(1, n_clusters)+1)

            # 红、橙、黄、绿、天蓝、青、紫、粉、棕、金、青绿、紫蓝、银白、黑、淡黄(偏白)、橄榄、
            custom_colors = ['#FF0000','#FFA500','#FFFF00','#008000',
                             '#00BFFF','#00FFFF','#800080','#FFC0CB',
                             '#A52A2A','#FFD700','#00FF7F','#7B68EE',
                             '#C0C0C0','#000000','#FFF8DC','#808000']
            print("labels:", labels)

            # 5. 按类别绘制峰值点
            plotted_labels = set()  # 用于跟踪哪些类别的图例已经添加
            for i in range(len(valid_peaks)):
                peak_index = valid_peaks[i]
                label = labels[i]
                # color = cmap(label % cmap.N)  # 使用模运算确保颜色索引在范围内 (虽然 get_cmap 通常处理得很好)
                color = custom_colors[label % len(custom_colors)]

                # 为每个类别只添加一次图例标签
                label_text = f'类别 {label}'
                if label not in plotted_labels:
                    ax.plot(peak_index, first_n_points[peak_index], 'o',
                            color=color, label=label_text)
                    plotted_labels.add(label)
                else:
                    # 后续同类别的点不再添加图例标签
                    ax.plot(peak_index, first_n_points[peak_index], 'o', color=color)

            # 6. 设置图形属性
            ax.set_title(f'{self.function_selector.currentText()} 聚类分类的结果',
                         fontproperties=get_chinese_font())
            ax.set_xlabel("时间点",fontproperties=get_chinese_font())  # 或者根据你的数据是秒还是索引
            ax.set_ylabel(ylabel if ylabel else "幅值",fontproperties=get_chinese_font())  # 使用传入的 Y 轴标签

            # 7. 添加图例
            ax.legend(fontsize=12,
                      prop=get_chinese_font())

            # 8. 调整布局防止标签重叠
            self.figure.tight_layout()

            # 9. !!! 关键：刷新 Canvas !!!
            self.canvas.draw()
            print("Matplotlib 画布已更新。")

        except Exception as e:
            print(f"更新绘图时出错: {e}")
            import traceback
            traceback.print_exc()
            # 可以在画布上显示错误信息
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'绘图错误:\n{e}', horizontalalignment='center', verticalalignment='center', color='red')
            self.canvas.draw()

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
        """示例：获取参数并准备运行算法"""
        # 1. 获取选中的算法名称
        selected_algorithm_name = self.function_selector.currentText()
        if not selected_algorithm_name:
            print("错误：没有选中任何算法。")
            # 可以添加 QMessageBox 提示用户
            return

        print(f"选定算法: {selected_algorithm_name}")

        # 2. 获取当前算法的参数
        parameters = self.get_current_parameters()
        if parameters is None:
            print("错误：无法获取当前算法的参数。")
            # 可以添加 QMessageBox 提示用户
            return

        print(f"获取参数: {parameters}")

        # 3. 从 DataManager 获取用于聚类的数据
        # !!! 关键：你需要确定从哪个 submission 获取数据 !!!
        #    通常是基于 submission_list_widget 的当前选中项
        from PyQt5.QtWidgets import QMessageBox
        current_item = self.submission_list_widget.currentItem()
        if not current_item:
            print("错误：请在'已提交的识别数据'中选择要进行聚类的数据。")
            # 使用 QMessageBox 提示用户
            QMessageBox.warning(self, "缺少数据", "请在'已提交的识别数据'中选择要进行聚类的数据项。")
            return

        selected_data_name = current_item.text()
        print(f"目标数据: {selected_data_name}")

        # 调用 DataManager 的方法来获取与该名称对应的数据
        # !!! 你需要在 DataManager 中实现 get_data_by_name 方法 !!!
        path_to_process, data_to_process = self.data_manager.get_data_by_name(selected_data_name)

        if data_to_process is None:
            print(f"错误：无法从 DataManager 获取名为 '{selected_data_name}' 的数据。")
            # 可以添加 QMessageBox 提示用户
            QMessageBox.critical(self, "数据错误", f"无法找到或加载名为 '{selected_data_name}' 的数据。")
            return

        # 假设 data_to_process 是一个适合聚类算法的格式，例如 numpy 数组
        print(f"成功获取数据，{getattr(data_to_process, 'shape', '未知')}")  # 打印数据形状（如果是数组）

        # 4. 根据算法名称调用不同的算法逻辑
        try:
            if selected_algorithm_name == "K-Means":
                # 调用你的 K-Means 算法实现
                # 假设你有 Algorithms.run_kmeans(data, params)
                from algorithms import Algorithms  # 确保导入
                print("调用 KMeans 算法...")
                self.signal_for_plot,self.peaks_for_plot,self.results = Algorithms.run_kmeans(path_to_process, data_to_process, parameters)

            elif selected_algorithm_name == "DBSCAN":
                # 调用你的 DBSCAN 算法实现
                # 假设你有 Algorithms.run_dbscan(data, params)
                from algorithms import Algorithms  # 确保导入
                print("调用 DBSCAN 算法...")
                # self.results = Algorithms.run_dbscan(path_to_process,data_to_process, parameters)

            elif selected_algorithm_name == "功能 3":
                print("功能 3 不需要执行复杂算法或尚未实现。")
                # 可以直接进行一些简单操作或显示提示
                self.results = "功能 3 执行完成"  # 示例结果

            else:
                print(f"错误：未知的算法或未处理的算法分支: {selected_algorithm_name}")
                QMessageBox.warning(self, "未实现", f"算法 '{selected_algorithm_name}' 的执行逻辑尚未实现。")
                return

            # 5. 处理和显示结果
            if self.results is not None:
                print(f"算法执行成功！结果预览（前10个）: {self.results[:10] if isinstance(self.results, (list, np.ndarray)) else self.results}")
                # 在这里更新你的主图区域 (self.main_plot_area) 来显示聚类结果
                self.update_plot(self.signal_for_plot, self.peaks_for_plot, self.results, "电流/NA")
                # self.update_plot(data_to_process, self.results)  # 调用一个专门的绘图函数
            elif isinstance(self.results, str):  # 处理 "功能 3" 等简单结果
                print(self.results)
                # 可以清除或更新绘图区显示文字信息
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, self.results, horizontalalignment='center', verticalalignment='center')
                self.canvas.draw()
            else:
                print("算法执行完成，但没有返回有效的聚类标签。")
                # 可以清除或更新绘图区显示提示
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, '算法未返回有效结果', horizontalalignment='center', verticalalignment='center')
                self.canvas.draw()


        except ImportError:
            print("错误：未能导入 algorithms 模块或其中包含的算法库（如 sklearn）。请确保已安装。")
            QMessageBox.critical(self, "导入错误", "运行算法所需的库未能导入，请检查安装。")
        except Exception as e:
            # 捕获算法执行中可能出现的任何其他错误
            print(f"运行算法 '{selected_algorithm_name}' 时出错: {e}")
            import traceback
            traceback.print_exc()  # 打印详细的错误堆栈
            QMessageBox.critical(self, "算法错误", f"执行 '{selected_algorithm_name}' 时发生错误:\n{e}")

        print("--- 聚类算法运行结束 ---")
