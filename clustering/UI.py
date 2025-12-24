"""
Clustering UI 模块：定义聚类页面的用户界面
"""
import os
import platform
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSizePolicy, QHBoxLayout, QPushButton, QStackedWidget,
    QComboBox, QFrame, QSplitter, QListWidget, QScrollArea
)
from PyQt5.QtCore import Qt
from matplotlib import font_manager
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from parameter_widgets import (
    BaseParameterWidget, KMeansParameterWidget,
    DBSCANParameterWidget, PlaceholderParameterWidget
)


def get_chinese_font():
    system = platform.system()
    if system == "Windows":
        # Windows 常见中文字体
        font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
    elif system == "Darwin":
        # macOS 使用 STHeiti 或 PingFang
        font_path = "/System/Library/Fonts/STHeiti Light.ttc"

    return font_manager.FontProperties(fname=font_path)


class ClusteringUI:
    """聚类页面的UI构建类"""

    def __init__(self, parent):
        """
        初始化聚类页面UI

        Args:
            parent: 父级组件，通常是 ClusteringPage 实例
        """
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        """设置UI界面"""
        main_layout = QHBoxLayout(self.parent)  # 主窗口水平布局

        # --- 左侧面板 (垂直 QSplitter 分隔) ---
        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.setObjectName("left_splitter")

        # --- 配置数据列表区 ---
        self.setup_data_list_area(left_splitter)

        # --- 下拉列表切换区 ---
        self.setup_dropdown_area(left_splitter)  # 封装下拉列表设置

        # --- 参数设置区 (现在包含 QStackedWidget) ---
        self.setup_parameter_area(left_splitter)  # 封装参数区设置

        # --- 按钮区域 ---
        self.button = QPushButton("运行聚类")
        self.button.setMaximumHeight(50)
        left_splitter.addWidget(self.button)

        self.save_button = QPushButton("保存结果")
        self.save_button.setMaximumHeight(50)
        left_splitter.addWidget(self.save_button)


        self.save_model_button = QPushButton("保存模型 (用于推测)")  # 新增
        self.save_model_button.setMaximumHeight(50)
        left_splitter.addWidget(self.save_model_button)

        # 设置左侧 Splitter 的初始大小 (大致比例)
        left_splitter.setSizes([150, 80, 300, 10, 10, 10])

        # 使用主 Splitter 统一 1:5 比例
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_splitter)

        # --- 设置绘图区域 ---
        # 注意：setup_plot_area 接受 parent_layout，这里我们传入 splitter (兼容 addWidget)
        self.setup_plot_area(main_splitter)

        main_splitter.setSizes([400, 2000])
        main_splitter.setCollapsible(0, False)

        main_layout.addWidget(main_splitter)

    def setup_data_list_area(self, parent_splitter: QSplitter):
        """配置数据加载区域的 UI 元素"""
        self.data_load_area = QWidget()
        self.data_load_area.setObjectName("data_load_area")
        data_load_layout = QVBoxLayout(self.data_load_area)
        data_load_layout.setContentsMargins(5, 5, 5, 5)
        data_load_layout.setSpacing(5)

        # 1. 标题标签
        data_load_label = QLabel("已提交的识别数据")
        data_load_label.setAlignment(Qt.AlignCenter)
        data_load_layout.addWidget(data_load_label)

        # 2. 列表控件，用于显示 submission 名称
        self.submission_list_widget = QListWidget()
        self.submission_list_widget.setObjectName("submissionList")
        data_load_layout.addWidget(self.submission_list_widget)

        # 样式表
        self.data_load_area.setStyleSheet('''
            QWidget#data_load_area {
                background-color: #f5f5f5;
                border: 1px solid #999;
                border-radius: 4px;
            }
            QLabel {
                font-size: 25px;
                color: #333;
                padding-bottom: 5px;
                font-weight: bold;
            }
            QListWidget#submissionList {
                background-color: white;
                border: 1px solid #ccc;
                font-size: 25px;
            }
            QListWidget#submissionList::item:selected {
                background-color: #e0e0e0;
                color: black;
            }
        ''')

        parent_splitter.addWidget(self.data_load_area)

    def setup_dropdown_area(self, parent_splitter: QSplitter):
        """设置下拉列表切换区域"""
        # 下拉列表切换区
        self.dropdown_switch_area = QWidget()
        self.dropdown_switch_area.setObjectName("dropdown_switch_area")
        # 使用 Frame 增加边框感
        dropdown_frame = QFrame(self.dropdown_switch_area)
        dropdown_frame.setFrameShape(QFrame.StyledPanel)

        dropdown_area_layout = QVBoxLayout(dropdown_frame)
        dropdown_area_layout.setContentsMargins(5, 5, 5, 5)

        # 定义下拉选项和映射
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
            QWidget#dropdown_switch_area { 
                 padding: 0px; 
            }
            QFrame { 
                 background-color: #e8e8e8;
                 border: 1px solid #bbbbbb;
                 border-radius: 3px;
            }
            QLabel { margin-bottom: 2px; font-size: 25px; color: #333;}
            QComboBox#function_selector { min-height: 22px; }
        ''')
        # 固定下拉区域的高度，使其紧凑
        self.dropdown_switch_area.setFixedHeight(self.dropdown_switch_area.sizeHint().height() + 5)
        parent_splitter.addWidget(self.dropdown_switch_area)

    def setup_parameter_area(self, parent_splitter: QSplitter):
        """设置参数区域"""
        # 参数设置区
        self.parameter_settings_area = QWidget()
        self.parameter_settings_area.setObjectName("parameter_settings_area")
        parameter_settings_layout = QVBoxLayout(self.parameter_settings_area)
        parameter_settings_layout.setContentsMargins(5, 5, 5, 5)
        parameter_settings_layout.setSpacing(0)

        self.parameter_stack = QStackedWidget()  # 创建 Stacked Widget
        parameter_settings_layout.addWidget(self.parameter_stack)

        # 实例化每个参数控件并添加到 Stacked Widget 和字典中
        self.param_widgets = {}  # 用于存储参数控件实例的字典
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
            QLabel { font-size: 24px; color: #555; }
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
        self.scroll_content_widget = QWidget()
        self.scroll_content_widget.setObjectName("scroll_content_widget")

        # 3. 为内部容器创建一个垂直布局
        self.plot_layout = QVBoxLayout(self.scroll_content_widget)
        self.plot_layout.setContentsMargins(5, 5, 5, 5)  # 设置边距
        self.plot_layout.setSpacing(10)  # 设置图形之间的间距

        # 4. 将内部容器设置为 QScrollArea 的控件
        self.scroll_area.setWidget(self.scroll_content_widget)

        # 5. 将 QScrollArea 添加到父布局
        parent_layout.addWidget(self.scroll_area)

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
