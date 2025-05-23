import sys
import os
import numpy as np
import time # 用于性能调试（可选）

from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QListWidget, QFileDialog,
    QStackedWidget, QSplitter, QTabWidget, QSpinBox, QFormLayout, QGroupBox,
    QSizePolicy, QComboBox, QListWidgetItem, QButtonGroup
)
from PyQt5.QtCore import Qt, QPointF, QTimer, center  # QTimer 用于延迟更新，避免过于频繁的重绘
import pyqtgraph as pg

from data_manager import *
from analysis import AnalysisPage
from clustering import ClusteringPage
from predict import PredictPage
from settings import SettingsPage

# 保持 pyqtgraph 配置
pg.setConfigOptions(useOpenGL=True)
pg.setConfigOptions(antialias=True)
# 增加全局降采样配置 (也可以在 plot 时单独设置)
# pg.setConfigOption('downsample', 5) # 默认降采样级别，可以根据需要调整

try:
    import pyabf
except ImportError:
    pyabf = None

def resource_path(relative_path):
    """打包后能正确找到资源文件"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class DataViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.data_manager = DataManager()
        self.analysis_page = AnalysisPage(self.data_manager)
        self.pages = {}

        self.data_file_paths = []  # 存放路径
        self.index = 0
        self.data = None # 存储完整的 (x, y) 数据
        self.full_x = None # 单独存储完整的 x 数据，方便查找索引
        self.full_y = None # 单独存储完整的 y 数据
        self.main_curve = None # 主图当前显示的曲线对象
        self.thumbnail_curve = None # 缩略图的曲线对象
        self.updating_range_programmatically = False # 用于防止信号循环触发的标志
        self.data_length = 0

        self.color_item_pen = '#000000'
        self.color_line = '#0000ff'

        self.UI()

    def UI(self):
        self.setWindowTitle("Nanopores tool")
        self.resize(2400, 1600)
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget) # 主窗口使用水平布局
        self.setCentralWidget(main_widget)

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
        self.refresh_list()
        data_load_layout.addWidget(self.file_list)

        self.import_button.clicked.connect(self.import_data_file)
        self.delete_button.clicked.connect(self.delete_selected_file)
        self.file_list.itemClicked.connect(self.load_selected_file)

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
        self.apply_range_button.clicked.connect(self.apply_data_range_to_view)
        func1_layout.addWidget(self.apply_range_button)
        # func1_layout.addStretch() # 在末尾添加弹性，让上面控件都往上挤
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

        self.color_line_combo.lineEdit().textChanged.connect(self.update_color_line_preview)

        # 添加到布局
        func2_layout.addWidget(color_line_label)
        line_layout = QHBoxLayout()
        line_layout.addWidget(self.color_line_combo)
        line_layout.addWidget(self.color_line_preview)
        func2_layout.addLayout(line_layout)

        # # ------- 添加颜色选择器：color_item_pen -------
        # color_item_label = QLabel("标记颜色 (color_item_pen):")
        # self.color_item_combo = QComboBox()
        # self.color_item_combo.setEditable(True)
        # self.color_item_combo.addItems(["#000000", "#ff0000", "#00ff00", "#0000ff", "#ff00ff"])
        # self.color_item_combo.setEditText("#ff0000")  # 默认值
        # self.color_item_preview = QLabel()
        # self.color_item_preview.setFixedSize(40, 20)
        # self.color_item_preview.setStyleSheet(
        #     f"background-color: {self.color_item_combo.currentText()}; border: 1px solid gray;")
        #
        # self.color_item_combo.lineEdit().textChanged.connect(self.update_color_item_preview)


        # 添加到布局
        # func2_layout.addWidget(color_item_label)
        # item_layout = QHBoxLayout()
        # item_layout.addWidget(self.color_item_combo)
        # item_layout.addWidget(self.color_item_preview)
        # func2_layout.addLayout(item_layout)

        self.function_area2.setLayout(func2_layout)
        self.function_area2.setStyleSheet('''
            QWidget#function_area2 {
                background-color: rgb(240, 240, 240);
                border: 1px solid black;
            }
        ''')
        # self.function_area2.setMinimumHeight(200)
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
        # self.path_input = QLineEdit()
        # self.load_button = QPushButton("加载数据")
        # self.load_button.clicked.connect(self.load_data)
        # top_toolbar.addWidget(self.path_input)
        # top_toolbar.addWidget(self.load_button)

        self.page_tabs = QTabWidget()
        self.main_plot_widget = pg.PlotWidget(name="MainPlot")
        self.main_plot_widget.setBackground('w')
        self.main_plot_widget.getPlotItem().getAxis('left').setPen(self.color_item_pen)
        self.main_plot_widget.getPlotItem().getAxis('bottom').setPen(self.color_item_pen)
        self.main_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.main_plot_widget.setClipToView(True)

        page1 = QWidget()
        layout1 = QVBoxLayout()
        layout1.addWidget(self.main_plot_widget)
        page1.setLayout(layout1)
        self.page_tabs.addTab(page1, "数据图")

        page2 = QLabel("这是第二页...")
        page2.setAlignment(Qt.AlignCenter)
        self.page_tabs.addTab(page2, "其他功能")

        self.coord_label = QLabel("坐标: N/A")
        self.coord_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        center_layout.addLayout(top_toolbar)
        center_layout.addWidget(self.page_tabs)
        center_layout.addWidget(self.coord_label)

        # 主布局
        internal_splitter = QSplitter(Qt.Horizontal)
        internal_splitter.addWidget(left_splitter)
        internal_splitter.addWidget(center_widget)
        internal_splitter.setSizes([480, 1920])


        # 区域选择控件
        self.region = pg.LinearRegionItem()
        self.region.setBrush([0, 0, 255, 50])
        self.region.setZValue(10)
        self.thumbnail_plot_widget.addItem(self.region, ignoreBounds=True)

        # 信号连接
        self.region.sigRegionChanged.connect(self.on_region_changed)
        self.main_plot_widget.sigXRangeChanged.connect(self.on_main_xrange_changed)
        self.main_plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)

        # 延迟更新的定时器
        self.update_timer = QTimer()
        self.update_timer.setInterval(50)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.actually_update_main_plot)

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

        self.button_view_predict = QPushButton()  # 对应第四个页面
        self.button_view_predict.setIcon(QIcon(resource_path("media/推理.svg")))
        self.button_view_predict.setToolTip('预测')
        self.button_view_predict.setCheckable(True)

        self.button_view_settings = QPushButton()  # 对应第五个页面
        self.button_view_settings.setIcon(QIcon(resource_path("media/设置.svg")))
        self.button_view_settings.setToolTip('设置')
        self.button_view_settings.setCheckable(True)

        # --- 按钮组实现互斥 ---
        self.toolbar_button_group = QButtonGroup(self)
        self.toolbar_button_group.addButton(self.button_view_data, 0)  # 关联索引 0
        self.toolbar_button_group.addButton(self.button_view_analysis, 1)  # 关联索引 1
        self.toolbar_button_group.addButton(self.button_view_clustering, 2)  # 关联索引 1
        self.toolbar_button_group.addButton(self.button_view_predict, 3)  # 关联索引 2
        self.toolbar_button_group.addButton(self.button_view_settings, 4)  # 关联索引 2
        self.toolbar_button_group.setExclusive(True)

        # --- 连接按钮点击信号到槽函数 ---
        self.button_view_data.clicked.connect(self.show_data_view_page)
        self.button_view_analysis.clicked.connect(self.show_analysis_page)
        self.button_view_clustering.clicked.connect(self.show_clustering_page)
        self.button_view_predict.clicked.connect(self.show_predict_page)
        self.button_view_settings.clicked.connect(self.show_settings_page)

        right_toolbar_layout.addWidget(self.button_view_data)
        right_toolbar_layout.addWidget(self.button_view_analysis)
        right_toolbar_layout.addWidget(self.button_view_clustering)
        right_toolbar_layout.addWidget(self.button_view_predict)
        right_toolbar_layout.addWidget(self.button_view_settings)
        # right_toolbar_layout.addStretch() # 移除拉伸，让按钮在顶部

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

        font_size = "25px"
        self.setStyleSheet(f"""
                ClusteringPage {{ /* 应用于整个页面，可以设置一个基础字体 */
                    /* font-size: {font_size}; */ /* 取消注释以设置全局默认值 */
                }}
                QLabel {{
                    font-size: {font_size};
                    /* 可以添加其他样式，如 color, padding 等 */
                }}
                QPushButton {{
                    font-size: {font_size};
                    padding: 5px 10px; /* 增加按钮内边距，让文字不拥挤 */
                }}
                QComboBox {{
                    font-size: {font_size};
                }}
                QListWidget {{
                    font-size: {font_size};
                }}
                /* 如果使用了 QGroupBox 或 QFrame 作为容器，也可以设置它们的标题字体 */
                QGroupBox::title {{
                    font-size: {font_size};
                    font-weight: bold;
                }}
                /* 对于参数设置区域内的控件 (假设它们也在 QLabel, QSpinBox 等内) */
                /* BaseParameterWidget QLabel, BaseParameterWidget QSpinBox, ... 如果想单独控制 */
                /* 但通常上面的全局设置就足够了 */
            """)


    def refresh_list(self):
        self.file_list.clear()
        self.data_file_paths.clear()
        for path in self.data_manager.get_all_files():
            self.data_file_paths.append(path)
            item = QListWidgetItem(os.path.basename(path))
            item.setToolTip(path)
            self.file_list.addItem(item)


    def import_data_file(self):
        # files, _ = QFileDialog.getOpenFileNames(
        #     self, "选择数据文件", "", "数据文件 (*.csv *.txt *.abf);;所有文件 (*)")
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择数据文件", "", "ABF 文件 (*.abf)")
        if not files:
            return

        for i, filepath in enumerate(files):
            if filepath not in self.data_file_paths:

                self.data_manager.add_file(filepath)

                self.data_file_paths.append(filepath)
                self.refresh_list()
                # self.show_analysis_page()
                self.analysis_page.refresh_list()  # 切换前刷新分析页列表

                # 如果是第一个文件，立即加载
                if len(self.data_file_paths) == 1 and i == 0:
                    self.load_data(filepath)
                print(len(self.data_file_paths))

    def delete_selected_file(self):
        current_item = self.file_list.currentItem()
        if current_item:
            row = self.file_list.row(current_item)

            if 0 <= row < len(self.data_file_paths):
                removed_path = self.data_file_paths.pop(row)
                self.data_manager.remove_file(removed_path)
                print(self.data_manager.get_all_files())
                self.refresh_list()
                self.analysis_page.refresh_list()  # 切换前刷新分析页列表

                print(f"已移除路径: {removed_path}")

            # 强制取消选中项，避免误触发
            self.file_list.clearSelection()

            # 若当前文件正是被删除的那个，且列表已空，清空数据
            if len(self.data_file_paths) == 0:
                self.clear_plots()
                self.label_count.setText("未加载")
                self.data = None
                self.full_x = None
                self.full_y = None
                self.data_length = 0
            elif self.data_file_paths:
                self.load_data(self.data_file_paths[0])

    def load_selected_file(self, item):
        self.index = self.file_list.row(item)
        print(self.file_list.row(item))
        # print(type(self.file_list.row(item)))
        filepath = self.data_file_paths[self.index]
        print(filepath)
        self.load_data(filepath)

    def load_data(self, filepath):
        print("调用加载函数")
        if filepath:
            # self.path_input.setText(filepath)
            ext = os.path.splitext(filepath)[-1].lower()
            print(f"开始加载 {filepath}...")
            load_start_time = time.time()

            try:
                # 清空旧数据和绘图
                self.clear_plots()
                self.data = None
                self.full_x = None
                self.full_y = None
                self.data_length = 0

                if ext == ".abf":
                    # ... (abf 加载逻辑保持不变) ...
                    if pyabf is None:
                        self.show_error("错误", "未安装 pyabf 库...")
                        return
                    abf = pyabf.ABF(filepath)
                    abf.setSweep(0)
                    x = abf.sweepX
                    y = abf.sweepY
                # elif ext in [".csv", ".txt"]:
                #     # ... (csv/txt 加载逻辑保持不变) ...
                #     try:
                #         # 使用 numpy.loadtxt 可能会对超大文件造成内存问题
                #         # 考虑使用 pandas 分块读取或更内存高效的方式
                #         # 这里暂时保持 numpy.loadtxt
                #         print("使用 numpy.loadtxt 加载，大文件可能需要较长时间和较多内存...")
                #         data_load = np.loadtxt(filepath, delimiter=',', comments='#')
                #     except ValueError:
                #         data_load = np.loadtxt(filepath, comments='#')
                #     except MemoryError:
                #         self.show_error("内存错误", "加载文件时内存不足，请尝试使用更小的数据集或优化加载方式。")
                #         return
                #
                #     if data_load.ndim == 1:
                #         x = np.arange(len(data_load))
                #         y = data_load
                #     elif data_load.shape[1] >= 2:
                #         x = data_load[:, 0]
                #         y = data_load[:, 1]
                #         if data_load.shape[1] > 2:
                #             print(f"警告: 文件包含超过2列，仅使用前两列。")
                #     else:
                #         raise ValueError("数据格式无法解析为 X, Y 列")
                else:
                    self.show_error("错误", f"不支持的文件类型: {ext}")
                    return

                load_end_time = time.time()
                print(f"文件加载完成，耗时: {load_end_time - load_start_time:.2f} 秒")

                # 存储完整数据
                self.full_x = x
                self.full_y = y
                self.data_length = len(x)
                self.data = (self.full_x, self.full_y) # self.data 仍然保留，可能有用

                # 更新UI信息
                self.label_count.setText(f"{self.data_length:,}") # 使用千位分隔符
                self.spin_start.setMaximum(self.data_length - 1 if self.data_length > 0 else 0)
                self.spin_end.setMaximum(self.data_length - 1 if self.data_length > 0 else 0)
                self.spin_start.setValue(0)
                self.spin_end.setValue(self.data_length - 1 if self.data_length > 0 else 0)

                # --- 设置绘图 ---
                self.setup_plots_after_load()

            except Exception as e:
                self.show_error("数据加载/处理失败", f"处理文件 '{os.path.basename(filepath)}' 时出错:\n{e}")
                self.clear_plots()

    # 更新预览函数
    def update_color_line_preview(self):
        print("调用数据点颜色变换")
        color = self.color_line_combo.currentText()
        self.color_line_preview.setStyleSheet(f"background-color: {color}; border: 1px solid gray;")
        self.color_line = color  # 绑定到变量
        self.load_data(self.data_file_paths[self.index])

    # # 更新预览函数
    # def update_color_item_preview(self):
    #     print("调用坐标轴颜色变换")
    #     color = self.color_item_combo.currentText()
    #     self.color_item_preview.setStyleSheet(f"background-color: {color}; border: 1px solid gray;")
    #     self.color_item_pen = color  # 绑定到变量
    #     self.load_data(self.data_file_paths[self.index])


    def setup_plots_after_load(self):
        """数据加载完成后，设置缩略图和主图的初始状态 (使用峰值保持降采样)"""
        if self.full_x is None or self.data_length == 0:
            print("没有加载数据，跳过绘图设置。")
            return

        print("开始设置绘图 (峰值保持降采样)...")
        setup_start_time = time.time()

        # --- 1. 计算完整数据范围 ---
        if not isinstance(self.full_x, np.ndarray): self.full_x = np.array(self.full_x)
        if not isinstance(self.full_y, np.ndarray): self.full_y = np.array(self.full_y)
        x_min, x_max = self.full_x[0], self.full_x[-1]
        print("正在计算完整 Y 轴范围...")
        y_range_start = time.time()
        try:
            y_min = np.nanmin(self.full_y)
            y_max = np.nanmax(self.full_y)
        except TypeError:
             print("警告：无法计算 Y 轴范围。使用默认范围 [0, 1]。")
             y_min, y_max = 0, 1
        y_range_end = time.time()
        print(f"Y 轴范围计算耗时: {y_range_end - y_range_start:.2f}秒")
        print(f"完整数据范围: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}]")


        # --- 2. 准备缩略图控件 ---
        self.thumbnail_plot_widget.clear()
        self.thumbnail_plot_widget.disableAutoRange()


        # --- 3. 执行峰值保持降采样并绘制 ---
        thumbnail_target_points = 1000 # 目标“区间”数量，绘制点数约为2倍
        thumbnail_sampling_step = max(1, self.data_length // thumbnail_target_points)
        num_intervals = self.data_length // thumbnail_sampling_step
        print(f"缩略图降采样步长 (区间大小): {thumbnail_sampling_step}, 区间数量: {num_intervals}")

        # 预分配数组以提高效率
        x_thumb_display = np.zeros(num_intervals * 2) # 每个区间画两个X点（起始点）
        y_thumb_display = np.zeros(num_intervals * 2) # 每个区间画Y的min和max

        valid_points = 0 # 记录实际添加了多少点对
        downsample_start_time = time.time()
        for i in range(num_intervals):
            start = i * thumbnail_sampling_step
            # 最后一个区间可能不足一个step，确保end不超过总长度
            end = min(start + thumbnail_sampling_step, self.data_length)
            if start >= end: # 防止空区间
                continue

            # 获取当前区间的数据
            y_interval = self.full_y[start:end]

            # 计算区间内的min和max
            interval_min = np.nanmin(y_interval)
            interval_max = np.nanmax(y_interval)

            # 使用区间的起始X值作为这两个点的X坐标
            x_coord = self.full_x[start]

            # 添加 (x, min) 和 (x, max) 到数组
            # 按 (x, min), (x, max), (x_next, min_next), (x_next, max_next)... 排列
            # 这样绘制时会画出垂直线段代表每个区间的范围
            idx = valid_points * 2
            x_thumb_display[idx] = x_coord
            y_thumb_display[idx] = interval_min
            x_thumb_display[idx + 1] = x_coord
            y_thumb_display[idx + 1] = interval_max
            valid_points += 1

        # 截取实际使用的部分
        x_thumb_display = x_thumb_display[:valid_points * 2]
        y_thumb_display = y_thumb_display[:valid_points * 2]

        downsample_end_time = time.time()
        print(f"峰值保持降采样计算耗时: {downsample_end_time - downsample_start_time:.3f}秒")
        print(f"在缩略图上绘制 {len(x_thumb_display)} 个点 (代表 {valid_points} 个区间)。")

        self.thumbnail_curve = self.thumbnail_plot_widget.plot(
            x_thumb_display, y_thumb_display,
            pen=pg.mkPen(color='gray', width=0.5)
        )

        # --- 4. 显式设置缩略图的范围 ---
        print(f"设置缩略图范围: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}] (Y轴边距 0.1)")
        self.thumbnail_plot_widget.setXRange(x_min, x_max, padding=0)
        self.thumbnail_plot_widget.setYRange(y_min, y_max, padding=0.1)

        # --- 5. 验证缩略图范围 ---
        vb_thumb = self.thumbnail_plot_widget.getViewBox()
        current_x_range = vb_thumb.viewRange()[0]
        current_y_range = vb_thumb.viewRange()[1]
        print(f"缩略图实际视图范围: X={current_x_range}, Y={current_y_range}")

        # --- 6. 添加 LinearRegionItem ---
        self.thumbnail_plot_widget.addItem(self.region, ignoreBounds=True)

        # --- 7. 设置主图的初始视图 ---
        # ... (这部分代码保持不变) ...
        initial_view_ratio = 0.01
        initial_x_end = x_min + (x_max - x_min) * initial_view_ratio
        initial_x_end = min(initial_x_end, x_max)
        print(f"设置主图初始 X 轴范围: [{x_min}, {initial_x_end}]")
        self.updating_range_programmatically = True
        self.region.setRegion([x_min, initial_x_end])
        self.main_plot_widget.setYRange(y_min, y_max, padding=0.1)
        self.main_plot_widget.setXRange(x_min, initial_x_end, padding=0)
        self.updating_range_programmatically = False

        setup_end_time = time.time()
        print(f"绘图设置完成，耗时: {setup_end_time - setup_start_time:.2f}秒")

    def on_region_changed(self):
        """当缩略图区域改变时调用 (用户拖动/调整区域)"""
        if self.updating_range_programmatically or self.full_x is None:
            return
        # 设置主图的X范围，这将触发 on_main_xrange_changed
        minX, maxX = self.region.getRegion()
        # print(f"Region changed: Setting main XRange to [{minX}, {maxX}]")
        # self.updating_range_programmatically = True # 标记是程序设置的范围
        self.main_plot_widget.setXRange(minX, maxX, padding=0, update=False) # 暂时不触发更新
        # self.updating_range_programmatically = False
        # 使用定时器延迟更新，避免拖动时频繁绘制
        self.update_timer.start()


    def on_main_xrange_changed(self):
        """当主图X范围改变时调用 (用户缩放/平移主图 或 程序设置范围)"""
        if self.full_x is None:
            return

        # 1. 更新缩略图的 Region
        if not self.updating_range_programmatically: # 只有当用户交互导致范围改变时才更新region
            self.updating_range_programmatically = True # 防止 setRegion 再次触发自己
            minX, maxX = self.main_plot_widget.getViewBox().viewRange()[0]
            # print(f"Main XRange changed: Setting region to [{minX}, {maxX}]")
            self.region.setRegion([minX, maxX])
            self.updating_range_programmatically = False

        # 2. 使用定时器延迟更新主图数据，避免过于频繁的计算和绘制
        self.update_timer.start()

    def actually_update_main_plot(self):
        """实际执行主图数据更新和重绘的函数 (使用峰值保持降采样)"""
        if self.full_x is None or self.data_length == 0:
            print("无数据，跳过主图更新")
            return

        # 获取当前主图的X轴可视范围
        x_min_view, x_max_view = self.main_plot_widget.getViewBox().viewRange()[0]
        update_start_time = time.time()

        # --- 查找数据索引 ---
        start_index = np.searchsorted(self.full_x, x_min_view, side='left')
        end_index = np.searchsorted(self.full_x, x_max_view, side='right')
        start_index = max(0, start_index - 1)  # 包含左边界附近的一个点
        end_index = min(self.data_length, end_index + 1)  # 包含右边界附近的一个点

        if start_index >= end_index:
            print("警告：计算出的索引范围无效，清空主图")
            if self.main_curve:
                self.main_plot_widget.removeItem(self.main_curve)
                self.main_curve = None
            return

        # 提取需要显示的数据子集 (仍然基于完整数据)
        x_subset = self.full_x[start_index:end_index]
        y_subset = self.full_y[start_index:end_index]
        points_in_subset = len(x_subset)
        # print(f"  Indices: [{start_index}, {end_index}], Points in subset: {points_in_subset}")

        # --- **修改部分：应用峰值保持降采样到主图** ---
        # 设定一个目标“区间”数量，例如主图窗口宽度的 1.5 到 2 倍
        # 目标点数约为 target_intervals * 2
        target_intervals_main = int(self.main_plot_widget.width() * 1.5)
        target_intervals_main = max(target_intervals_main, 250)  # 最少也计算几百个区间
        target_intervals_main = min(target_intervals_main, 25000)  # 设置一个上限

        if points_in_subset > target_intervals_main * 2:  # 只有当数据点显著多于目标点数时才降采样
            # 计算降采样步长 (每个区间包含多少个原始点)
            main_sampling_step = max(1, points_in_subset // target_intervals_main)
            # 计算实际能生成的区间数量
            num_actual_intervals = points_in_subset // main_sampling_step

            # print(f"  主图降采样: {points_in_subset} 点 -> {num_actual_intervals} 区间 (步长 {main_sampling_step})")

            # 预分配数组
            x_display = np.zeros(num_actual_intervals * 2)
            y_display = np.zeros(num_actual_intervals * 2)
            valid_points_main = 0

            downsample_start_time = time.time()
            # 在 subset 上执行 min/max 采样
            for i in range(num_actual_intervals):
                # 注意：这里的 start_sub, end_sub 是相对于 subset 的索引
                start_sub = i * main_sampling_step
                end_sub = min(start_sub + main_sampling_step, points_in_subset)
                if start_sub >= end_sub: continue

                # 获取当前区间的数据 (从 subset 中获取)
                y_sub_interval = y_subset[start_sub:end_sub]

                # 计算区间内的min和max
                # 使用 try-except 捕获可能的空区间或全NaN区间 (虽然理论上不应发生)
                try:
                    interval_min = np.nanmin(y_sub_interval)
                    interval_max = np.nanmax(y_sub_interval)
                except ValueError:  # 如果区间为空或全为NaN
                    continue  # 跳过这个区间

                # 使用区间的起始X值 (从 subset 中获取)
                x_coord = x_subset[start_sub]

                # 添加 (x, min) 和 (x, max) 到数组
                idx = valid_points_main * 2
                x_display[idx] = x_coord
                y_display[idx] = interval_min
                x_display[idx + 1] = x_coord
                y_display[idx + 1] = interval_max
                valid_points_main += 1

            # 截取实际使用的部分
            x_display = x_display[:valid_points_main * 2]
            y_display = y_display[:valid_points_main * 2]
            # downsample_end_time = time.time()
            # print(f"  主图峰值降采样计算耗时: {downsample_end_time - downsample_start_time:.4f}秒")
            # print(f"  绘制 {len(x_display)} 个点 (代表 {valid_points_main} 个区间)")

        else:
            # 如果子集中的点数不多，直接绘制所有点
            # print(f"  无需降采样主图子集 ({points_in_subset} points).")
            x_display = x_subset
            y_display = y_subset

        points_to_plot = len(x_display)
        # print(f"  Points to plot: {points_to_plot}")

        # --- 绘制 ---
        if self.main_curve:
            # 优化：如果曲线对象已存在，尝试用 setData 更新，可能比移除再添加更快
            # 注意：setData 期望 x 和 y 长度一致，我们的 min/max 方法保证了这一点
            try:
                update_plot_start = time.time()
                self.main_curve.setData(x=x_display, y=y_display)
                # print(f"    setData took: {time.time() - update_plot_start:.4f} s")
            except Exception as e:
                # 如果 setData 失败（例如对象被意外删除），则回退到移除和添加
                # print(f"    setData failed ({e}), falling back to remove/add.")
                self.main_plot_widget.removeItem(self.main_curve)
                self.main_curve = self.main_plot_widget.plot(x_display, y_display, pen=self.color_line, name="Data")
        else:
            # 如果曲线不存在，则创建
            self.main_curve = self.main_plot_widget.plot(x_display, y_display, pen=self.color_line, name="Data")

        # # --- 之前的绘制逻辑 (注释掉或删除) ---
        # if self.main_curve:
        #     self.main_plot_widget.removeItem(self.main_curve)
        #     self.main_curve = None
        # self.main_curve = self.main_plot_widget.plot(
        #     x_display, y_display,
        #     pen='blue', name="Data",
        # )

        update_end_time = time.time()


    def clear_plots(self):
        """清空主图和缩略图"""
        print("Clearing plots...")
        self.main_plot_widget.clear()
        self.thumbnail_plot_widget.clear()
        self.main_curve = None
        self.thumbnail_curve = None
        # 重新添加 region (如果它存在)
        if hasattr(self, 'region'):
             self.thumbnail_plot_widget.addItem(self.region, ignoreBounds=True)
        self.coord_label.setText("坐标: N/A")
        self.label_count.setText("未加载")
        self.spin_start.setValue(0)
        self.spin_end.setValue(0)
        self.spin_start.setMaximum(0)
        self.spin_end.setMaximum(0)


    def on_mouse_moved(self, event):
        """处理主图上的鼠标移动事件，显示坐标"""
        # (与上一版相同，但注意性能)
        if not self.main_plot_widget.plotItem.vb.sceneBoundingRect().contains(event):
             return # 鼠标不在绘图区

        vb = self.main_plot_widget.plotItem.vb
        mouse_point = vb.mapSceneToView(event)
        x_coord = mouse_point.x()
        y_coord = mouse_point.y()
        self.coord_label.setText(f"坐标: X={x_coord:.4f}, Y={y_coord:.4f}")


    def apply_data_range_to_view(self):
        """应用起始索引和结束索引设置到主图的视图范围"""
        if self.full_x is not None and self.data_length > 0:
            start_idx = self.spin_start.value()
            end_idx = self.spin_end.value()

            # 索引验证
            if start_idx >= self.data_length: start_idx = self.data_length - 1
            if end_idx >= self.data_length: end_idx = self.data_length - 1
            if start_idx < 0: start_idx = 0
            if end_idx < start_idx: end_idx = start_idx

            # 更新SpinBox的值为有效值
            self.spin_start.setValue(start_idx)
            self.spin_end.setValue(end_idx)

            # 获取对应X值
            start_x = self.full_x[start_idx]
            end_x = self.full_x[end_idx]

            # 如果起始和结束X值相同（例如只有一个点），稍微扩展范围
            if start_x == end_x:
                # 尝试向前和向后查找不同的X值，或设置一个最小宽度
                prev_idx = max(0, start_idx - 1)
                next_idx = min(self.data_length - 1, end_idx + 1)
                start_x = self.full_x[prev_idx]
                end_x = self.full_x[next_idx]
                if start_x == end_x: # 如果还是相同，给个默认小范围
                    delta = (self.full_x[-1] - self.full_x[0]) * 0.001 if self.data_length > 1 else 1
                    start_x -= delta/2
                    end_x += delta/2

            print(f"Applying index range [{start_idx}, {end_idx}] -> XRange [{start_x}, {end_x}]")

            self.updating_range_programmatically = True # 阻止信号循环
            # 更新主图范围，这将触发数据更新
            self.main_plot_widget.setXRange(start_x, end_x, padding=0)
            # 自动调整Y轴 (可选，但推荐)
            y_subset = self.full_y[start_idx : end_idx + 1]
            if len(y_subset) > 0:
                 self.main_plot_widget.setYRange(np.min(y_subset), np.max(y_subset), padding=0.1)

            # 更新缩略图区域
            self.region.setRegion([start_x, end_x])
            self.updating_range_programmatically = False

            # 手动触发一次更新以确保应用了范围
            self.actually_update_main_plot()

        else:
            self.show_error("错误", "没有加载数据，无法应用范围。")

    def show_error(self, title, message):
        """显示一个简单的错误消息框"""
        print(f"--- {title} ---")
        print(message)
        print("-----------------")

    # --- 新增：页面切换的槽函数 ---
    def show_data_view_page(self):
        self.main_stack.setCurrentIndex(0)
        print("切换到 数据视图 页面")

    def show_analysis_page(self):
        if 'analysis' not in self.pages:
            self.analysis_page = AnalysisPage(self.data_manager)
            self.analysis_page.request_refresh.connect(self.refresh_list)
            self.main_stack.addWidget(self.analysis_page)
            self.pages['analysis'] = self.analysis_page
        self.main_stack.setCurrentWidget(self.pages['analysis'])
        print("切换到 分析 页面")

    def show_clustering_page(self):
        if 'clustering' not in self.pages:
            self.clustering_page = ClusteringPage(self.data_manager)
            self.main_stack.addWidget(self.clustering_page)
            self.pages['clustering'] = self.clustering_page
        self.main_stack.setCurrentWidget(self.pages['clustering'])
        print("切换到 聚类 页面")

    def show_predict_page(self):
        if 'predict' not in self.pages:
            self.predict_page = PredictPage()
            self.main_stack.addWidget(self.predict_page)
            self.pages['predict'] = self.predict_page
        self.main_stack.setCurrentWidget(self.pages['predict'])
        print("切换到 预测 页面")

    def show_settings_page(self):
        if 'settings' not in self.pages:
            self.settings_page = SettingsPage()
            self.main_stack.addWidget(self.settings_page)
            self.pages['settings'] = self.settings_page
        self.main_stack.setCurrentWidget(self.pages['settings'])
        print("切换到 设置 页面")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = DataViewer()
    viewer.show()
    sys.exit(app.exec_())