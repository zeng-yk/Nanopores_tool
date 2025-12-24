"""
Training UI 模块：定义训练页面的用户界面
"""
import os
import sys
import platform
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, 
    QListWidget, QSplitter, QFrame, QComboBox, QDoubleSpinBox, 
    QSpinBox, QFormLayout, QScrollArea, QSizePolicy, QTextEdit,
    QStackedWidget, QLineEdit, QGroupBox, QGridLayout
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
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
        font_path = "C:/Windows/Fonts/simhei.ttf"
    elif system == "Darwin":
        font_path = "/System/Library/Fonts/STHeiti Light.ttc"
    else:
        font_path = "" # Fallback
    return font_manager.FontProperties(fname=font_path)

class TrainingUI:
    """训练页面的UI构建类"""

    def __init__(self, parent):
        """
        初始化训练页面UI
        Args:
            parent: 父级组件，通常是 TrainingPage 实例
        """
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        """设置UI界面"""
        main_layout = QHBoxLayout(self.parent)

        # --- 左侧控制面板 ---
        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.setObjectName("left_splitter")

        # 1. 训练数据选择区
        self.setup_data_selection_area(left_splitter)

        # 2. 模型配置区
        self.setup_model_config_area(left_splitter)

        # 3. 操作按钮与日志区
        self.setup_control_area(left_splitter)

        # 设置左侧 Splitter 比例 (调整以容纳两个列表)
        left_splitter.setSizes([400, 200, 200])
        
        # 使用主 Splitter 统一 1:5 比例
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_splitter)

        # --- 右侧绘图区 ---
        self.setup_plot_area(main_splitter)

        main_splitter.setSizes([400, 2000])
        main_splitter.setCollapsible(0, False)
        
        main_layout.addWidget(main_splitter)

    def setup_data_selection_area(self, parent_splitter):
        """配置数据选择区域"""
        area = QWidget()
        area.setObjectName("data_load")
        layout = QVBoxLayout(area)
        layout.setContentsMargins(5, 5, 5, 5)

        # --- A. 分类结果列表 ---
        csv_group = QGroupBox("1. 分类结果 (CSV)")
        csv_layout = QVBoxLayout(csv_group)
        csv_layout.setContentsMargins(5, 5, 5, 5)

        csv_btn_layout = QHBoxLayout()
        self.import_csv_btn = QPushButton()
        self.import_csv_btn.setIcon(QIcon(resource_path("media/导入.svg")))
        self.import_csv_btn.setToolTip("导入 CSV 文件")
        
        self.remove_csv_btn = QPushButton()
        self.remove_csv_btn.setIcon(QIcon(resource_path("media/文本剔除.svg")))
        self.remove_csv_btn.setToolTip("删除选中 CSV")
        
        csv_btn_layout.addWidget(self.import_csv_btn)
        csv_btn_layout.addWidget(self.remove_csv_btn)
        csv_btn_layout.addStretch()
        csv_layout.addLayout(csv_btn_layout)

        self.csv_list = QListWidget()
        self.csv_list.setSelectionMode(QListWidget.MultiSelection)
        csv_layout.addWidget(self.csv_list)
        
        layout.addWidget(csv_group)

        # --- B. 原始文件列表 ---
        abf_group = QGroupBox("2. 原始文件 (ABF)")
        abf_layout = QVBoxLayout(abf_group)
        abf_layout.setContentsMargins(5, 5, 5, 5)

        abf_btn_layout = QHBoxLayout()
        self.import_abf_btn = QPushButton()
        self.import_abf_btn.setIcon(QIcon(resource_path("media/导入.svg")))
        self.import_abf_btn.setToolTip("导入 ABF 文件")
        
        self.remove_abf_btn = QPushButton()
        self.remove_abf_btn.setIcon(QIcon(resource_path("media/文本剔除.svg")))
        self.remove_abf_btn.setToolTip("删除选中 ABF")
        
        abf_btn_layout.addWidget(self.import_abf_btn)
        abf_btn_layout.addWidget(self.remove_abf_btn)
        abf_btn_layout.addStretch()
        abf_layout.addLayout(abf_btn_layout)

        self.abf_list = QListWidget()
        self.abf_list.setSelectionMode(QListWidget.MultiSelection)
        abf_layout.addWidget(self.abf_list)

        layout.addWidget(abf_group)
        
        # 样式
        area.setStyleSheet('''
            QWidget#data_load {
                background-color: #f5f5f5;
                border: 1px solid #999;
                border-radius: 4px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 4px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QListWidget {
                border: 1px solid #ccc;
                background-color: white;
            }
        ''')
        parent_splitter.addWidget(area)

    def setup_model_config_area(self, parent_splitter):
        """配置模型参数区域"""
        area = QWidget()
        layout = QVBoxLayout(area)
        layout.setContentsMargins(5, 5, 5, 5)

        # 1. 标题 (稍微大一点点，设为 14px)
        title_label = QLabel("3. 模型配置:")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; font-family: Arial; color: #000;")
        layout.addWidget(title_label)

        # 2. 模型类型选择区域
        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(0, 0, 0, 0)

        lbl_model = QLabel("模型类型:")
        # 这里也设为小字体 12px
        lbl_model.setStyleSheet("font-size: 12px; font-weight: bold; font-family: Arial;")
        lbl_model.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        grid_layout.addWidget(lbl_model, 0, 0)

        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["BP", "SNN"])
        self.model_type_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.model_type_combo.setMinimumWidth(200)

        # --- 【重点修改 1】下拉框样式：默认灰字，悬停黑字 ---
        self.model_type_combo.setStyleSheet("""
            QComboBox {
                background-color: #e0e0e0; 
                color: #666666;  /* 默认：深灰色 */
                border: 1px solid #999;
                border-radius: 4px;
                padding: 5px;
                font-family: Arial;
                font-size: 12px;
            }
            QComboBox:hover {
                background-color: #d0d0d0; 
                color: #000000;  /* 悬停：黑色 */
                border: 1px solid #666;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 1px;
                border-left-color: #999;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: black;
                selection-background-color: #007bff;
                selection-color: white;
            }
        """)

        grid_layout.addWidget(self.model_type_combo, 0, 1)
        grid_layout.setColumnStretch(1, 1)
        layout.addLayout(grid_layout)

        # 3. 参数堆栈
        self.params_stack = QStackedWidget()

        # --- 辅助函数：添加行 (在这里强制修改字体大小) ---
        def add_grid_row(grid, row, label_text, widget):
            lbl = QLabel(label_text)
            # --- 【重点修改 2】强制指定字体为 12px 和 Arial ---
            lbl.setStyleSheet("font-size: 12px; font-weight: bold; font-family: Arial; color: #333;")
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

            # 同时把输入框的字体也设为 12px，保持对齐
            widget.setStyleSheet(
                "font-size: 12px; font-family: Arial; border: 1px solid #999; border-radius: 4px; padding: 2px;")

            grid.addWidget(lbl, row, 0)
            grid.addWidget(widget, row, 1)

        # --- BP 参数页 ---
        self.bp_params_page = QWidget()
        bp_grid = QGridLayout(self.bp_params_page)
        bp_grid.setContentsMargins(0, 0, 0, 0)

        self.bp_hidden_layers = QLineEdit("100, 50")
        self.bp_hidden_layers.setPlaceholderText("e.g. 100, 50")
        add_grid_row(bp_grid, 0, "隐藏层结构:", self.bp_hidden_layers)

        self.bp_learning_rate = QDoubleSpinBox()
        self.bp_learning_rate.setRange(0.0001, 1.0)
        self.bp_learning_rate.setSingleStep(0.001)
        self.bp_learning_rate.setValue(0.001)
        self.bp_learning_rate.setDecimals(4)
        self.bp_learning_rate.setAlignment(Qt.AlignLeft)
        add_grid_row(bp_grid, 1, "学习率 (LR):", self.bp_learning_rate)

        self.bp_epochs = QSpinBox()
        self.bp_epochs.setRange(1, 10000)
        self.bp_epochs.setValue(200)
        self.bp_epochs.setAlignment(Qt.AlignLeft)
        add_grid_row(bp_grid, 2, "训练轮数 (Epochs):", self.bp_epochs)

        self.bp_batch_size = QSpinBox()
        self.bp_batch_size.setRange(1, 2048)
        self.bp_batch_size.setValue(32)
        self.bp_batch_size.setAlignment(Qt.AlignLeft)
        add_grid_row(bp_grid, 3, "批次大小 (Batch):", self.bp_batch_size)

        bp_grid.setColumnStretch(1, 1)
        self.params_stack.addWidget(self.bp_params_page)

        # --- SNN 参数页 ---
        self.snn_params_page = QWidget()
        snn_grid = QGridLayout(self.snn_params_page)
        snn_grid.setContentsMargins(0, 0, 0, 0)

        self.snn_time_steps = QSpinBox()
        self.snn_time_steps.setRange(1, 1000)
        self.snn_time_steps.setValue(20)
        self.snn_time_steps.setAlignment(Qt.AlignLeft)
        add_grid_row(snn_grid, 0, "时间步长 (Steps):", self.snn_time_steps)

        self.snn_tau = QDoubleSpinBox()
        self.snn_tau.setRange(0.1, 10.0)
        self.snn_tau.setValue(2.0)
        self.snn_tau.setAlignment(Qt.AlignLeft)
        add_grid_row(snn_grid, 1, "膜时间常数 (Tau):", self.snn_tau)

        self.snn_learning_rate = QDoubleSpinBox()
        self.snn_learning_rate.setRange(0.0001, 1.0)
        self.snn_learning_rate.setSingleStep(0.001)
        self.snn_learning_rate.setValue(1e-3)
        self.snn_learning_rate.setDecimals(4)
        self.snn_learning_rate.setAlignment(Qt.AlignLeft)
        add_grid_row(snn_grid, 2, "学习率 (LR):", self.snn_learning_rate)

        self.snn_epochs = QSpinBox()
        self.snn_epochs.setRange(1, 10000)
        self.snn_epochs.setValue(100)
        self.snn_epochs.setAlignment(Qt.AlignLeft)
        add_grid_row(snn_grid, 3, "训练轮数 (Epochs):", self.snn_epochs)

        self.snn_batch_size = QSpinBox()
        self.snn_batch_size.setRange(1, 2048)
        self.snn_batch_size.setValue(32)
        self.snn_batch_size.setAlignment(Qt.AlignLeft)
        add_grid_row(snn_grid, 4, "批次大小 (Batch):", self.snn_batch_size)

        snn_grid.setColumnStretch(1, 1)
        self.params_stack.addWidget(self.snn_params_page)

        layout.addWidget(self.params_stack)

        # 4. 全局容器样式
        # 这里只保留容器的背景色和边框，具体的文字样式已经在上面针对性设置了
        area.setStyleSheet('''
            QWidget { 
                background-color: #f5f5f5; 
                border: 1px solid #999; 
                border-radius: 4px; 
            }
        ''')
        parent_splitter.addWidget(area)

    def setup_control_area(self, parent_splitter):
        """配置控制按钮和日志区域"""
        area = QWidget()
        layout = QVBoxLayout(area)
        layout.setContentsMargins(5, 5, 5, 5)

        layout.addWidget(QLabel("4. 操作:"))

        self.train_btn = QPushButton("开始训练")
        self.train_btn.setFixedHeight(40)
        layout.addWidget(self.train_btn)

        self.save_model_btn = QPushButton("保存模型")
        self.save_model_btn.setFixedHeight(40)
        self.save_model_btn.setEnabled(False) # 初始不可用
        layout.addWidget(self.save_model_btn)

        layout.addWidget(QLabel("训练日志:"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        area.setStyleSheet('''
            QWidget { background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px; }
            QLabel { font-weight: bold; font-size: 14px; }
            QPushButton { background-color: #007bff; color: white; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #0056b3; }
            QPushButton:disabled { background-color: #cccccc; }
            QTextEdit { background-color: white; font-family: monospace; }
        ''')
        parent_splitter.addWidget(area)

    def setup_plot_area(self, main_layout):
        """配置右侧绘图区域"""
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.plot_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)

        right_layout.addWidget(self.scroll_area)
        
        main_layout.addWidget(right_panel)

    def add_plot(self, figure):
        """添加 matplotlib 图表到右侧区域"""
        canvas = FigureCanvas(figure)
        canvas.setMinimumHeight(400)
        self.plot_layout.addWidget(canvas)

    def clear_plots(self):
        """清空绘图区域"""
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
