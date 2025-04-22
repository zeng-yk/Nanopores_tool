# parameter_widgets.py
from PyQt5.QtWidgets import (QWidget, QLabel, QSpinBox, QDoubleSpinBox,
                             QSlider, QCheckBox, QGridLayout, QVBoxLayout, QSpacerItem, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal

# (可选) 创建一个基础类，定义通用接口
class BaseParameterWidget(QWidget):
    # 定义一个信号，当任何参数值改变时发射（可选，但有用）
    parameters_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

    def get_parameters(self) -> dict:
        """获取当前控件设置的参数，返回一个字典。"""
        raise NotImplementedError("子类必须实现 get_parameters 方法")

    def connect_signals(self):
        """将内部控件的值变化信号连接到 parameters_changed 信号。"""
        # 子类中实现具体连接
        pass

# --- 功能 1 (例如 K-Means) 的参数控件 ---
class KMeansParameterWidget(BaseParameterWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QGridLayout(self) # 使用网格布局方便对齐标签和控件
        layout.setContentsMargins(10, 10, 10, 10) # 内边距
        layout.setSpacing(8) # 控件间距

        # 参数1: K 值
        self.k_label = QLabel("簇的数量 (K):")
        self.k_spinbox = QSpinBox()
        self.k_spinbox.setMinimum(2)
        self.k_spinbox.setMaximum(100) # 设置合理范围
        self.k_spinbox.setValue(3)    # 默认值
        layout.addWidget(self.k_label, 0, 0) # 第0行第0列
        layout.addWidget(self.k_spinbox, 0, 1) # 第0行第1列

        # 参数2: 最大迭代次数
        self.max_iter_label = QLabel("最大迭代次数:")
        self.max_iter_spinbox = QSpinBox()
        self.max_iter_spinbox.setMinimum(10)
        self.max_iter_spinbox.setMaximum(1000)
        self.max_iter_spinbox.setValue(300)
        layout.addWidget(self.max_iter_label, 1, 0)
        layout.addWidget(self.max_iter_spinbox, 1, 1)

        # 添加一个弹簧，将控件推到顶部
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer, 2, 0, 1, 2) # 跨越两列

        self.connect_signals() # 连接信号

    def get_parameters(self) -> dict:
        return {
            "n_clusters": self.k_spinbox.value(),
            "max_iter": self.max_iter_spinbox.value()
        }

    def connect_signals(self):
         self.k_spinbox.valueChanged.connect(self.parameters_changed.emit)
         self.max_iter_spinbox.valueChanged.connect(self.parameters_changed.emit)


# --- 功能 2 (例如 DBSCAN) 的参数控件 ---
class DBSCANParameterWidget(BaseParameterWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # 参数1: Epsilon (eps)
        self.eps_label = QLabel("邻域半径 (eps):")
        self.eps_spinbox = QDoubleSpinBox() # 使用 DoubleSpinBox 处理浮点数
        self.eps_spinbox.setDecimals(2)    # 显示两位小数
        self.eps_spinbox.setMinimum(0.01)
        self.eps_spinbox.setMaximum(10.0)
        self.eps_spinbox.setSingleStep(0.1)
        self.eps_spinbox.setValue(0.5)
        layout.addWidget(self.eps_label, 0, 0)
        layout.addWidget(self.eps_spinbox, 0, 1)

        # 参数2: MinPts
        self.min_pts_label = QLabel("核心点最小邻居数 (MinPts):")
        self.min_pts_spinbox = QSpinBox()
        self.min_pts_spinbox.setMinimum(2)
        self.min_pts_spinbox.setMaximum(100)
        self.min_pts_spinbox.setValue(5)
        layout.addWidget(self.min_pts_label, 1, 0)
        layout.addWidget(self.min_pts_spinbox, 1, 1)

        # 添加弹簧
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer, 2, 0, 1, 2)

        self.connect_signals()

    def get_parameters(self) -> dict:
        return {
            "eps": self.eps_spinbox.value(),
            "min_samples": self.min_pts_spinbox.value()
        }

    def connect_signals(self):
        self.eps_spinbox.valueChanged.connect(self.parameters_changed.emit)
        self.min_pts_spinbox.valueChanged.connect(self.parameters_changed.emit)

# --- 功能 3 (可以是一个更简单的，或占位符) ---
class PlaceholderParameterWidget(BaseParameterWidget):
     def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.label = QLabel("功能 3 无需参数设置")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

     def get_parameters(self) -> dict:
         return {} # 没有参数

     # connect_signals 不需要实现

# --- (可选) 算法逻辑模块 ---
# 你可以将实际的聚类算法放在单独的文件中，例如 algorithms.py
# class Algorithms:
#     @staticmethod
#     def run_kmeans(data, params):
#         from sklearn.cluster import KMeans
#         kmeans = KMeans(n_clusters=params['n_clusters'], max_iter=params['max_iter'], n_init=10)
#         labels = kmeans.fit_predict(data)
#         return labels
#
#     @staticmethod
#     def run_dbscan(data, params):
#         from sklearn.cluster import DBSCAN
#         dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
#         labels = dbscan.fit_predict(data)
#         return labels