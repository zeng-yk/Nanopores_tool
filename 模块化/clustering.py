# clustering.py
import threading

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy, QHBoxLayout, QPushButton, QStackedWidget, \
    QComboBox, QFrame, QSplitter, QListWidget, QApplication
from PyQt5.QtCore import Qt, pyqtSignal
from parameter_widgets import (BaseParameterWidget, KMeansParameterWidget,
                               DBSCANParameterWidget, PlaceholderParameterWidget)

class ClusteringPage(QWidget):

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
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
        self.setup_dropdown_area(left_splitter) # 封装下拉列表设置

        # 3. 参数设置区 (现在包含 QStackedWidget)
        self.setup_parameter_area(left_splitter) # 封装参数区设置


        # 设置左侧 Splitter 的初始大小 (大致比例)
        left_splitter.setSizes([150, 80, 300])

        # --- 右侧面板 (占位符) ---
        self.main_plot_area = QWidget()
        self.main_plot_area.setObjectName("main_plot_area")
        main_plot_layout = QVBoxLayout(self.main_plot_area)
        main_plot_label = QLabel("主图")
        main_plot_label.setAlignment(Qt.AlignCenter)
        main_plot_layout.addWidget(main_plot_label)
        # 基本样式
        self.main_plot_area.setStyleSheet('''
                    QWidget#main_plot_area {
                        background-color: #e0e0e0;
                        border: 1px solid #999999;
                    }
                     QLabel { font-size: 20px; color: #666; }
                ''')

        # --- 组合主布局 ---
        main_layout.addWidget(left_splitter)
        main_layout.addWidget(self.main_plot_area)

        # 设置主布局拉伸因子 (右侧更宽)
        main_layout.setStretchFactor(left_splitter, 1)
        main_layout.setStretchFactor(self.main_plot_area, 3)

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

        self.parameter_stack = QStackedWidget() # 创建 Stacked Widget
        parameter_settings_layout.addWidget(self.parameter_stack)

        # 实例化每个参数控件并添加到 Stacked Widget 和字典中
        for name, WidgetClass in self.algorithm_map.items():
            widget_instance = WidgetClass()
            self.parameter_stack.addWidget(widget_instance)
            self.param_widgets[name] = widget_instance # 存储实例引用
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
        selected_algorithm_name = self.function_selector.currentText()
        parameters = self.get_current_parameters()

        if parameters is None:
            print("无法获取当前算法的参数。")
            return

        print(f"准备运行算法: {selected_algorithm_name}")
        print(f"使用参数: {parameters}")