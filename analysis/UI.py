"""
Analysis UI 模块：定义分析页面的用户界面
"""
import os
import sys
import platform
import numpy as np

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QPushButton, QFileDialog, QHBoxLayout, QListWidgetItem,
    QSplitter, QFormLayout, QSpinBox, QColorDialog, QLabel, QDoubleSpinBox, QCheckBox, QMessageBox,
    QInputDialog, QSizePolicy, QButtonGroup, QApplication, QProgressDialog
)
from PyQt5.QtCore import Qt, QTimer

from matplotlib import font_manager
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

def resource_path(relative_path):
    """打包后能正确找到资源文件"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def get_chinese_font():
    system = platform.system()
    if system == "Windows":
        # Windows 常见中文字体
        font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
    elif system == "Darwin":
        # macOS 使用 STHeiti 或 PingFang
        font_path = "/System/Library/Fonts/STHeiti Light.ttc"

    return font_manager.FontProperties(fname=font_path)

class AnalysisUI:
    """分析页面的UI构建类"""
    def __init__(self, parent):
        """
        初始化分析页面UI

        Args:
            parent: 父级组件，通常是 AnalysisPage 实例
        """
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        """设置UI界面"""
        main_layout = QHBoxLayout(self.parent)  # 主窗口使用水平布局
        left_splitter = QSplitter(Qt.Vertical)

        # 数据加载区
        self.data_load = QWidget()
        self.data_load.setObjectName("data_load")
        data_load_layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.add_button = QPushButton()
        self.add_button.setIcon(QIcon(resource_path("media/导入.svg")))
        self.add_button.setToolTip("导入文件")

        self.remove_button = QPushButton()
        self.remove_button.setIcon(QIcon(resource_path("media/文本剔除.svg")))
        self.remove_button.setToolTip("删除选中项")

        btn_layout.addWidget(self.add_button)
        btn_layout.addWidget(self.remove_button)
        data_load_layout.addLayout(btn_layout)

        self.file_list = QListWidget()
        self.file_list.setObjectName("file_list")
        data_load_layout.addWidget(self.file_list)

        # 样式
        self.data_load.setStyleSheet('''
            QWidget#data_load {
                background-color: #f5f5f5;
                border: 1px solid #999;
                border-radius: 4px;
            }
            QListWidget {
                border: 1px solid #ccc;
            }
            QListWidget#file_list::item:selected {
                background-color: #e0e0e0;
                color: black;
            }
        ''')

        self.data_load.setLayout(data_load_layout)
        self.data_load.setMaximumHeight(700)
        left_splitter.addWidget(self.data_load)

        # 参数设置区
        self.parameter_settings = QWidget()
        self.parameter_settings.setObjectName("parameter_settings")
        parameter_settings_layout = QVBoxLayout(self.parameter_settings)
        self.parameter_settings.setStyleSheet('''
            QWidget#parameter_settings {
                background-color: #f5f5f5;
                border: 1px solid #999;
                margin-top: 10px;
                padding-top: 10px;
                margin-bottom: 10px;
            }
        ''')
        self.function_1 = QWidget()
        self.function_1.setObjectName("function_1")
        function_1_layout = QVBoxLayout()

        form = QFormLayout()

        # 复选框
        self.Downsampling_checkbox = QCheckBox("性能模式")
        self.Downsampling_checkbox.setChecked(True)  # 默认勾选

        self.btn1 = QPushButton("波峰")
        self.btn1.setMinimumHeight(50)
        self.btn2 = QPushButton("波谷")
        self.btn2.setMinimumHeight(50)
        # 设置为可点击状态
        self.btn1.setCheckable(True)
        self.btn2.setCheckable(True)
        # 创建互斥按钮组
        self.group = QButtonGroup(self.parent)
        self.group.setExclusive(True)  # 设置互斥
        self.group.addButton(self.btn1)
        self.group.addButton(self.btn2)

        Button_layout = QHBoxLayout()
        Button_layout.addWidget(self.btn1)
        Button_layout.addWidget(self.btn2)

        # 添加各个参数
        self.height_widget, self.cb_height, self.spin_height = self.add_checkbox_spinbox("height", 100, checked=False)
        self.threshold_widget, self.cb_threshold, self.spin_threshold = self.add_checkbox_spinbox("threshold", 0.5,
                                                                                                  decimals=True,
                                                                                                  checked=False)
        self.distance_widget, self.cb_distance, self.spin_distance = self.add_checkbox_spinbox("distance", 100)
        self.prominence_widget, self.cb_prominence, self.spin_prominence = self.add_checkbox_spinbox("prominence", 50.0,
                                                                                                     decimals=True)
        self.width_widget, self.cb_width, self.spin_width = self.add_checkbox_spinbox("width", 3, checked=False)

        form.addRow(self.add_label("Height:"), self.height_widget)
        form.addRow(self.add_label("Threshold:"), self.threshold_widget)
        form.addRow(self.add_label("Distance:"), self.distance_widget)
        form.addRow(self.add_label("Prominence:"), self.prominence_widget)
        form.addRow(self.add_label("Width:"), self.width_widget)

        self.color1_widget, self.btn_color1 = self.add_color_selector("#0000FF")
        self.color2_widget, self.btn_color2 = self.add_color_selector("#00FF00")
        self.color3_widget, self.btn_color3 = self.add_color_selector("#FFD700")

        form.addRow(self.add_label("原始信号:"), self.color1_widget)
        form.addRow(self.add_label("峰宽线:"), self.color2_widget)
        form.addRow(self.add_label("半峰宽:"), self.color3_widget)

        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        function_1_layout.addWidget(self.Downsampling_checkbox)
        function_1_layout.addLayout(Button_layout)
        function_1_layout.addLayout(form)

        self.apply_range_button = QPushButton("应用参数")
        self.apply_range_button.setMinimumHeight(50)
        function_1_layout.addWidget(self.apply_range_button)

        self.load_time_label = QLabel("加载时间：尚未加载")
        function_1_layout.addWidget(self.load_time_label)

        self.load_timer = QTimer()

        self.function_1.setLayout(function_1_layout)
        self.function_1.setStyleSheet('''
            QWidget#function_1 {
                border: 1px solid black;
            }
        ''')
        parameter_settings_layout.addWidget(self.function_1)

        parameter_settings_layout.addStretch(1)
        self.submit_button = QPushButton("提交识别结果")
        self.submit_button.setMinimumHeight(50)
        parameter_settings_layout.addWidget(self.submit_button)

        self.save_button = QPushButton("导出识别结果")
        self.save_button.setMinimumHeight(50)
        parameter_settings_layout.addWidget(self.save_button)
        parameter_settings_layout.addStretch(1)

        left_splitter.addWidget(self.parameter_settings)

        left_splitter.setSizes([600, 400])  # Example sizes

        # 右侧的图展示区域
        right_section_widget = QWidget()
        right_section_layout = QVBoxLayout(right_section_widget)

        # 主图区
        self.main_plot = QWidget()
        self.main_plot.setObjectName("main_plot")
        main_plot_layout = QVBoxLayout(self.main_plot)

        # 添加 Matplotlib Canvas
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        main_plot_layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.ax.tick_params(labelsize=14)

        self.main_plot.setStyleSheet('''
            QWidget#main_plot {
                background-color: #e0e0e0;
                border: 1px solid #999;
            }
        ''')
        right_section_layout.addWidget(self.main_plot)

        bottom_right_splitter = QSplitter(Qt.Horizontal)

        # 图1区 (Empty Placeholder with navigation)
        plot1_container_widget = QWidget()
        plot1_container_layout = QVBoxLayout(plot1_container_widget)
        plot1_container_layout.setContentsMargins(0, 0, 0, 0) 

        plot1_nav_layout = QHBoxLayout()
        plot1_nav_layout.addStretch()
        self.prev_plot1_button = QPushButton("<")
        self.next_plot1_button = QPushButton(">")

        plot1_nav_layout.addWidget(self.prev_plot1_button)
        plot1_nav_layout.addWidget(self.next_plot1_button)
        plot1_nav_layout.addStretch()

        self.plot1 = QWidget()
        self.plot1.setObjectName("plot1")
        self.plot1_layout = QVBoxLayout(self.plot1)
        self.plot1.setStyleSheet('''
            QWidget#plot1 {
                background-color: #d0d0d0;
                border: 1px solid #999;
            }
        ''')

        plot1_container_layout.addLayout(plot1_nav_layout)
        plot1_container_layout.addWidget(self.plot1)

        bottom_right_splitter.addWidget(plot1_container_widget)

        # 图1的数据区 (Empty Placeholder)
        self.plot1_data = QWidget()
        self.plot1_data.setObjectName("plot1_data")
        self.plot1_data_layout = QVBoxLayout(self.plot1_data)
        self.plot1_data.setStyleSheet('''
            QWidget#plot1_data {
                background-color: #d0d0d0;
                border: 1px solid #999;
            }
        ''')
        bottom_right_splitter.addWidget(self.plot1_data)

        bottom_right_splitter.setSizes([600, 300])

        right_section_layout.addWidget(bottom_right_splitter)
        right_section_layout.setStretchFactor(self.main_plot, 2)
        right_section_layout.setStretchFactor(bottom_right_splitter, 1)

        # 使用 QSplitter 替代原来的布局方式，统一 1:5 比例
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(right_section_widget)
        main_splitter.setSizes([400, 2000])
        main_splitter.setCollapsible(0, False)

        main_layout.addWidget(main_splitter)

    @staticmethod
    def add_label(name):
        label = QLabel(name)
        # label.setFixedWidth(100) # Removed fixed width for responsiveness
        return label

    @staticmethod
    # ====== 第一组：可选参数（带复选框）======
    def add_checkbox_spinbox(label_text, default_value=1.0, decimals=False, checked=True):
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        checkbox = QCheckBox()
        checkbox.setChecked(checked)

        # 使用浮点或整数 SpinBox
        if decimals:
            spinbox = QDoubleSpinBox()
            spinbox.setDecimals(2)
            spinbox.setSingleStep(0.1)
            spinbox.setRange(0.0, 1_000_000.0)  # 设置范围
        else:
            spinbox = QSpinBox()
            spinbox.setRange(0, 1_000_000)  # 设置范围
        spinbox.setValue(default_value)

        # 💡 让 SpinBox 左对齐 & 可拉伸
        spinbox.setAlignment(Qt.AlignLeft)
        spinbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        spinbox.setMinimumWidth(80)

        # 自定义值记录器，None 表示未启用
        spinbox.logical_value = default_value if checked else None
        spinbox.setEnabled(checked)

        def on_checkbox_state_changed(state):
            if state == Qt.Checked:
                spinbox.setEnabled(True)
                spinbox.logical_value = spinbox.value()
            else:
                spinbox.setEnabled(False)
                spinbox.logical_value = None

        checkbox.stateChanged.connect(on_checkbox_state_changed)

        def on_spinbox_value_changed(val):
            if checkbox.isChecked():
                spinbox.logical_value = val

        spinbox.valueChanged.connect(on_spinbox_value_changed)

        layout.addWidget(checkbox)
        layout.addWidget(spinbox)

        layout.setStretch(1, 1)  # SpinBox 铺满右侧

        return container, checkbox, spinbox

    @staticmethod
    # ====== 第二组：颜色参数（无复选框）======
    def add_color_selector(default_color="#FF0000"):
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel()
        label.setFixedWidth(150)
        label.setStyleSheet(f"background-color: {default_color}; border: 1px solid #666;")

        button = QPushButton("选择")
        button.setMinimumHeight(45)
        button.setMaximumWidth(100)

        def choose_color():
            color = QColorDialog.getColor()
            if color.isValid():
                label.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #666;")
                button.color = color.name()

        button.clicked.connect(choose_color)
        button.color = default_color  # 存储颜色值

        layout.addWidget(label)
        layout.addWidget(button)
        return container, button
