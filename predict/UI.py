from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QFrame, QVBoxLayout, QLabel, QListWidget, QPushButton, QWidget, QScrollArea, \
    QSplitter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import platform
from matplotlib import font_manager

def get_chinese_font():
    system = platform.system()
    if system == "Windows":
        # Windows 常见中文字体
        font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
    elif system == "Darwin":
        # macOS 使用 STHeiti 或 PingFang
        font_path = "/System/Library/Fonts/STHeiti Light.ttc"

    return font_manager.FontProperties(fname=font_path)

class PredictUI:
    """推测页面的UI构建类"""

    def __init__(self, parent):
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout(self.parent)

        # --- 左侧控制面板 ---
        left_panel = QFrame()
        left_panel.setObjectName("left_panel")
        left_panel.setStyleSheet('''
            QFrame#left_panel {
                background-color: #f5f5f5;
                border: 1px solid #999;
                border-radius: 4px;
            }
            QLabel {
                font-weight: bold;
                font-size: 14px;
            }
            QListWidget {
                border: 1px solid #ccc;
                background-color: white;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        ''')
        left_layout = QVBoxLayout(left_panel)

        # 1. 模型选择区
        left_layout.addWidget(QLabel("1. 选择训练好的模型:"))
        self.model_list = QListWidget()
        left_layout.addWidget(self.model_list)

        # 2. 数据选择区
        left_layout.addWidget(QLabel("2. 选择待分析数据:"))
        self.data_list = QListWidget()
        left_layout.addWidget(self.data_list)

        # 3. 操作按钮
        self.run_btn = QPushButton("运行推测")
        self.run_btn.setFixedHeight(40)
        left_layout.addWidget(self.run_btn)

        self.save_btn = QPushButton("保存推测结果")
        self.save_btn.setFixedHeight(40)
        left_layout.addWidget(self.save_btn)

        left_layout.addStretch()  # 弹簧

        # --- 右侧绘图区 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.plot_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)

        right_layout.addWidget(self.scroll_area)

        # 组合
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 2000])
        splitter.setCollapsible(0, False)

        main_layout.addWidget(splitter)

    # 辅助方法：添加 matplotlib 画布 (与 ClusteringUI 类似)
    def add_plot(self, figure):
        canvas = FigureCanvas(figure)
        canvas.setMinimumHeight(400)
        self.plot_layout.addWidget(canvas)

    def clear_plots(self):
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()