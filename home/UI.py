import os
import sys

from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QListWidget, QFileDialog,
    QStackedWidget, QSplitter, QTabWidget, QSpinBox, QFormLayout, QGroupBox,
    QSizePolicy, QComboBox, QListWidgetItem, QButtonGroup
)
from PyQt5.QtCore import Qt, QPointF, QTimer, center
import pyqtgraph as pg

def resource_path(relative_path):
    """打包后能正确找到资源文件"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class DataViewerUI:
    """数据查看器的UI构建类"""
    def __init__(self, main_window):
        self.main_window = main_window
        self.setup_ui()

    def setup_ui(self):
        """设置UI界面"""
        self.main_window.setWindowTitle("Nanopores tool")
        self.main_window.resize(2400, 1600)
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget) # 主窗口使用水平布局
        self.main_window.setCentralWidget(main_widget)

        # === 1. 创建 QStackedWidget 用于页面切换 ===
        self.main_stack = QStackedWidget()

        # === 2. 创建第一个页面 (数据视图) ===
        data_view_page = QWidget()
        data_view_page_layout = QHBoxLayout(data_view_page)  # 此页面的布局

        # 左侧布局：功能区 + 缩略图
        left_splitter = QSplitter(Qt.Vertical)

        # 数据加载区
        self.data_load = QWidget()
        self.data_load.setObjectName("data_load")
        data_load_layout = QVBoxLayout()

        # 按钮区域（水平排列）
        btn_layout = QHBoxLayout()
        self.import_button = QPushButton()
        self.import_button.setIcon(QIcon(resource_path("media/导入.svg")))  # 你可以换成你自己的图片
        self.import_button.setToolTip("导入文件")

        self.delete_button = QPushButton()
        self.delete_button.setIcon(QIcon(resource_path("media/文本剔除.svg")))
        self.delete_button.setToolTip("删除选中项")

        btn_layout.addWidget(self.import_button)
        btn_layout.addWidget(self.delete_button)
        data_load_layout.addLayout(btn_layout)

        # 文件列表
        self.file_list = QListWidget()
        self.file_list.setObjectName("file_list")
        data_load_layout.addWidget(self.file_list)

        # 设置布局
        self.data_load.setLayout(data_load_layout)
        self.data_load.setMaximumHeight(700)  # 可根据需求设置

        # 样式
        self.data_load.setStyleSheet('''
            QWidget#data_load {
                background-color: #f5f5f5;
                border: 1px solid #999;
            }
            QListWidget {
                border: 1px solid #ccc;
            }
            QListWidget#file_list::item:selected {
                background-color: #e0e0e0; /* 可以改为灰色或移除 */
                color: black;
            }
        ''')

        left_splitter.addWidget(self.data_load)

        # 功能区1
        self.function_area1 = QWidget()
        self.function_area1.setObjectName("function_area1")
        func1_layout = QVBoxLayout()
        func1_layout.addStretch(1)

        form = QFormLayout()
        self.label_count = QLabel("未加载")
        self.spin_start = QSpinBox()
        self.spin_end = QSpinBox()
        self.spin_start.setMinimum(0)
        self.spin_end.setMinimum(0)
        self.spin_start.setMaximum(2147483647)
        self.spin_end.setMaximum(2147483647)
        form.addRow("数据总数：", self.label_count)
        form.addRow("起始索引：", self.spin_start)
        form.addRow("结束索引：", self.spin_end)
        func1_layout.addLayout(form)
        func1_layout.addStretch(1)

        self.apply_range_button = QPushButton("应用范围到主图")
        func1_layout.addWidget(self.apply_range_button)
        func1_layout.addStretch(1)

        self.function_area1.setLayout(func1_layout)
        self.function_area1.setStyleSheet('''
            QWidget#function_area1 {
                background-color: rgb(240, 240, 240);
                border: 1px solid black;
            }
            QSpinBox {
                background-color: rgb(240, 240, 240);
                border: 1px solid gray;
                padding: 3px;
                border-radius: 8px;
            }
        ''')
        self.function_area1.setMaximumHeight(300)


        # 功能区2
        self.function_area2 = QWidget()
        self.function_area2.setObjectName("function_area2")
        func2_layout = QVBoxLayout()

        # ------- 添加颜色选择器：color_line -------
        color_line_label = QLabel("数据点颜色:")
        self.color_line_combo = QComboBox()
        self.color_line_combo.setEditable(True)
        self.color_line_combo.addItems(["#000000", "#ff0000", "#00ff00", "#0000ff", "#ffff00"])
        self.color_line_combo.setEditText("#0000ff")  # 默认值
        self.color_line_preview = QLabel()
        self.color_line_preview.setFixedSize(40, 20)
        self.color_line_preview.setStyleSheet(
            f"background-color: {self.color_line_combo.currentText()}; border: 1px solid gray;")

        # 添加到布局
        func2_layout.addWidget(color_line_label)
        line_layout = QHBoxLayout()
        line_layout.addWidget(self.color_line_combo)
        line_layout.addWidget(self.color_line_preview)
        func2_layout.addLayout(line_layout)

        self.function_area2.setLayout(func2_layout)
        self.function_area2.setStyleSheet('''
            QWidget#function_area2 {
                background-color: #f5f5f5;
                border: 1px solid #999;
                border-radius: 4px;
            }
        ''')
        self.function_area2.setMaximumHeight(300)

        left_splitter.addWidget(self.function_area1)
        left_splitter.addWidget(self.function_area2)

        # 缩略图
        self.thumbnail_plot_widget = pg.PlotWidget(name="Thumbnail")
        self.thumbnail_plot_widget.setBackground('w')
        self.thumbnail_plot_widget.showAxis('left', False)
        self.thumbnail_plot_widget.showAxis('bottom', False)
        self.thumbnail_plot_widget.setMenuEnabled(False)
        self.thumbnail_plot_widget.setMouseEnabled(x=False, y=False)
        left_splitter.addWidget(self.thumbnail_plot_widget)
        left_splitter.setSizes([500,100,100,300])

        # 右侧：顶部工具栏 + 标签页 + 坐标标签
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)

        top_toolbar = QHBoxLayout()

        self.page_tabs = QTabWidget()
        self.main_plot_widget = pg.PlotWidget(name="MainPlot")
        self.main_plot_widget.setBackground('w')
        self.main_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.main_plot_widget.setClipToView(True)

        page1 = QWidget()
        layout1 = QVBoxLayout()
        layout1.addWidget(self.main_plot_widget)
        page1.setLayout(layout1)
        self.page_tabs.addTab(page1, "数据图")

        # === 第二页：统计与直方图 ===
        self.stats_page = QWidget()
        self.stats_layout = QVBoxLayout(self.stats_page)

        # 1. 统计信息栏
        stats_info_layout = QHBoxLayout()
        self.lbl_mean = QLabel("均值: N/A")
        self.lbl_std = QLabel("标准差: N/A")
        self.lbl_min = QLabel("最小值: N/A")
        self.lbl_max = QLabel("最大值: N/A")
        
        # 设置字体大小
        for lbl in [self.lbl_mean, self.lbl_std, self.lbl_min, self.lbl_max]:
            lbl.setStyleSheet("font-size: 16px; margin-right: 15px;")

        stats_info_layout.addWidget(self.lbl_mean)
        stats_info_layout.addWidget(self.lbl_std)
        stats_info_layout.addWidget(self.lbl_min)
        stats_info_layout.addWidget(self.lbl_max)
        stats_info_layout.addStretch()

        self.btn_refresh_stats = QPushButton("刷新统计")
        self.btn_refresh_stats.setToolTip("基于当前主图视图范围计算统计数据")
        stats_info_layout.addWidget(self.btn_refresh_stats)

        self.stats_layout.addLayout(stats_info_layout)

        # 2. 直方图
        self.histogram_plot = pg.PlotWidget(name="Histogram")
        self.histogram_plot.setBackground('w')
        self.histogram_plot.setTitle("幅值直方图 (当前视图)", color='k', size='12pt')
        self.histogram_plot.setLabel('left', '计数', color='k')
        self.histogram_plot.setLabel('bottom', '幅值', color='k')
        self.histogram_plot.showGrid(x=True, y=True, alpha=0.3)
        self.stats_layout.addWidget(self.histogram_plot)

        self.page_tabs.addTab(self.stats_page, "统计分析")

        self.coord_label = QLabel("坐标: N/A")
        self.coord_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        center_layout.addLayout(top_toolbar)
        center_layout.addWidget(self.page_tabs)
        center_layout.addWidget(self.coord_label)

        # 主布局
        internal_splitter = QSplitter(Qt.Horizontal)
        internal_splitter.addWidget(left_splitter)
        internal_splitter.addWidget(center_widget)
        internal_splitter.setSizes([400, 2000])
        internal_splitter.setCollapsible(0, False)

        # 区域选择控件
        self.region = pg.LinearRegionItem()
        self.region.setBrush([0, 0, 255, 50])
        self.region.setZValue(10)
        self.thumbnail_plot_widget.addItem(self.region, ignoreBounds=True)

        # === 3. 将你原来的主界面 (由 internal_splitter 管理) 添加到数据视图页面 ===
        data_view_page_layout.addWidget(internal_splitter)

        # === 4. 将数据视图页面添加到 QStackedWidget ===
        self.main_stack.addWidget(data_view_page)

        # === 5. 创建右侧垂直功能按钮栏 ===
        right_toolbar_widget = QWidget()
        right_toolbar_widget.setObjectName("right_toolbar")
        right_toolbar_layout = QVBoxLayout(right_toolbar_widget)
        right_toolbar_layout.setContentsMargins(5, 10, 5, 10)
        right_toolbar_layout.setSpacing(10)
        right_toolbar_layout.setAlignment(Qt.AlignTop)  # 按钮顶部对齐

        self.button_view_data = QPushButton()  # 对应第一个页面
        self.button_view_data.setIcon(QIcon(resource_path("media/折线图.svg")))
        self.button_view_data.setToolTip('数据视图')
        self.button_view_data.setCheckable(True)
        self.button_view_data.setChecked(True)  # 默认选中

        self.button_view_analysis = QPushButton()  # 对应第二个页面
        self.button_view_analysis.setIcon(QIcon(resource_path("media/分析.svg")))
        self.button_view_analysis.setToolTip('波形分析')
        self.button_view_analysis.setCheckable(True)

        self.button_view_clustering = QPushButton()  # 对应第三个页面
        self.button_view_clustering.setIcon(QIcon(resource_path("media/聚类.svg")))
        self.button_view_clustering.setToolTip('聚类')
        self.button_view_clustering.setCheckable(True)

        self.button_view_train = QPushButton()  # 对应第四个页面
        self.button_view_train.setIcon(QIcon(resource_path("media/模型训练.svg")))
        self.button_view_train.setToolTip('训练')
        self.button_view_train.setCheckable(True)

        self.button_view_predict = QPushButton()  # 对应第五个页面
        self.button_view_predict.setIcon(QIcon(resource_path("media/推理.svg")))
        self.button_view_predict.setToolTip('预测')
        self.button_view_predict.setCheckable(True)

        self.button_view_settings = QPushButton()  # 对应第六个页面
        self.button_view_settings.setIcon(QIcon(resource_path("media/设置.svg")))
        self.button_view_settings.setToolTip('设置')
        self.button_view_settings.setCheckable(True)

        # --- 按钮组实现互斥 ---
        self.toolbar_button_group = QButtonGroup(self.main_window)
        self.toolbar_button_group.addButton(self.button_view_data, 0)  # 关联索引 0
        self.toolbar_button_group.addButton(self.button_view_analysis, 1)  # 关联索引 1
        self.toolbar_button_group.addButton(self.button_view_clustering, 2)  # 关联索引 2
        self.toolbar_button_group.addButton(self.button_view_train, 3)  # 关联索引 3
        self.toolbar_button_group.addButton(self.button_view_predict, 4)  # 关联索引 4
        self.toolbar_button_group.addButton(self.button_view_settings, 5)  # 关联索引 5
        self.toolbar_button_group.setExclusive(True)

        right_toolbar_layout.addWidget(self.button_view_data)
        right_toolbar_layout.addWidget(self.button_view_analysis)
        right_toolbar_layout.addWidget(self.button_view_clustering)
        right_toolbar_layout.addWidget(self.button_view_train)
        right_toolbar_layout.addWidget(self.button_view_predict)
        right_toolbar_layout.addWidget(self.button_view_settings)

        right_toolbar_widget.setFixedWidth(70)  # 设置固定宽度
        right_toolbar_widget.setStyleSheet('''
            QWidget#right_toolbar {
                background-color: #e8e8e8; /* 淡灰色背景 */
                border-left: 1px solid #c0c0c0; /* 左边框线 */
            }
            QPushButton {
                padding: 12px 6px;
                text-align: center;
                border: 1px solid #b0b0b0;
                background-color: #f5f5f5;
                border-radius: 4px;
                min-height: 35px;
                font-size: 10pt; /* 稍大字体 */
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                border-color: #909090;
            }
            QPushButton:checked { /* 选中状态 */
                background-color: #d0e0f0; /* 选中颜色 */
                border: 1px solid #a0c0e0;
                font-weight: bold;
                color: #1a1a1a; /* 深色字体 */
            }
            QPushButton:pressed {
                background-color: #c8c8c8;
            }
        ''')

        # === 6. 将 QStackedWidget 和 右侧工具栏 添加到主窗口布局 ===
        main_layout.addWidget(self.main_stack, 1)  # StackedWidget 占据主要空间 (拉伸因子=1)
        main_layout.addWidget(right_toolbar_widget, 0)  # 右侧工具栏宽度固定 (拉伸因子=0)

        # 设置全局样式
        font_size = "25px"
        self.main_window.setStyleSheet(f"""
            QLabel {{
                font-size: {font_size};
            }}
            QPushButton {{
                font-size: {font_size};
                padding: 5px 10px;
            }}
            QComboBox {{
                font-size: {font_size};
            }}
            QListWidget {{
                font-size: {font_size};
            }}
            QGroupBox::title {{
                font-size: {font_size};
                font-weight: bold;
            }}
        """)

        # 延迟更新的定时器
        self.update_timer = QTimer()
        self.update_timer.setInterval(50)
        self.update_timer.setSingleShot(True)