# analysis_page.py
import os
import time

import pyabf
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QListWidget, QPushButton, QFileDialog, QHBoxLayout, QListWidgetItem, \
    QSplitter, QFormLayout, QSpinBox, QColorDialog, QLabel, QDoubleSpinBox, QCheckBox, QMessageBox, QInputDialog
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from matplotlib import font_manager

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths
from PyQt5.QtCore import QThread
from load_worker import LoadWorker  # 导入多线程
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


class AnalysisPage(QWidget):
    request_refresh = pyqtSignal()

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)

        self.filepath = None

        self.data_manager = data_manager
        self.width_widget = None
        self.prominence_widget = None
        self.distance_widget = None
        self.threshold_widget = None
        self.height_widget = None

        self.index = 0
        self.peaks = None
        self.prominences = None
        self.height = None
        self.width = None
        self.peak_properties = None

        self.data = None
        self.full_x = None
        self.full_y = None

        # 文件列表
        self.data_file_paths = []  # 存放路径
        self.ui()

        self.flag = True

    def ui(self):
        main_layout = QHBoxLayout(self)  # 主窗口使用水平布局
        left_splitter = QSplitter(Qt.Vertical)

        # 数据加载区 (Existing code)
        self.data_load = QWidget()
        self.data_load.setObjectName("data_load")
        data_load_layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.add_button = QPushButton()
        self.add_button.setIcon(QIcon("media/导入.svg"))
        self.add_button.setToolTip("导入文件")

        self.remove_button = QPushButton()
        self.remove_button.setIcon(QIcon("media/文本剔除.svg"))
        self.remove_button.setToolTip("删除选中项")

        btn_layout.addWidget(self.add_button)
        btn_layout.addWidget(self.remove_button)
        data_load_layout.addLayout(btn_layout)

        self.file_list = QListWidget()
        self.refresh_list()
        data_load_layout.addWidget(self.file_list)

        self.add_button.clicked.connect(self.add_file)
        self.remove_button.clicked.connect(self.remove_file)
        self.file_list.itemClicked.connect(self.load_selected_file)

        # 样式
        self.data_load.setStyleSheet('''
                    QWidget#data_load {
                        background-color: #f5f5f5;
                        border: 1px solid #999;
                    }
                    QListWidget {
                        border: 1px solid #ccc;
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
            }
        ''')
        self.function_1 = QWidget()
        self.function_1.setObjectName("function_1")
        function_1_layout = QVBoxLayout()

        form = QFormLayout()

        # 添加各个参数
        self.height_widget, self.cb_height, self.spin_height = self.add_checkbox_spinbox("height", 100, checked=False)
        self.threshold_widget, self.cb_threshold, self.spin_threshold = self.add_checkbox_spinbox("threshold", 0.5,
                                                                                                  decimals=True,
                                                                                                  checked=False)
        self.distance_widget, self.cb_distance, self.spin_distance = self.add_checkbox_spinbox("distance", 100)
        self.prominence_widget, self.cb_prominence, self.spin_prominence = self.add_checkbox_spinbox("prominence", 50.0,
                                                                                                     decimals=True)
        self.width_widget, self.cb_width, self.spin_width = self.add_checkbox_spinbox("width", 3, checked=False)

        form.addRow("Height:", self.height_widget)
        form.addRow("Threshold:", self.threshold_widget)
        form.addRow("Distance:", self.distance_widget)
        form.addRow("Prominence:", self.prominence_widget)
        form.addRow("Width:", self.width_widget)

        self.color1_widget, self.btn_color1 = self.add_color_selector("#FF0000")
        self.color2_widget, self.btn_color2 = self.add_color_selector("#00FF00")
        self.color3_widget, self.btn_color3 = self.add_color_selector("#0000FF")

        form.addRow("Color 1:", self.color1_widget)
        form.addRow("Color 2:", self.color2_widget)
        form.addRow("Color 3:", self.color3_widget)

        function_1_layout.addLayout(form)

        self.apply_range_button = QPushButton("应用参数")
        self.apply_range_button.clicked.connect(self.apply_data_range_to_view)
        function_1_layout.addWidget(self.apply_range_button)

        self.load_time_label = QLabel("加载时间：尚未加载")
        function_1_layout.addWidget(self.load_time_label)

        self.load_timer = QTimer()
        self.load_timer.timeout.connect(self.update_load_time)
        self.load_start_time = None

        self.function_1.setLayout(function_1_layout)
        self.function_1.setStyleSheet('''
            QWidget#function_1 {
                border: 1px solid black;
            }
        ''')
        parameter_settings_layout.addWidget(self.function_1)

        parameter_settings_layout.addStretch(1)
        self.submit_button = QPushButton("提交识别结果")
        self.submit_button.clicked.connect(self.submit_data)
        parameter_settings_layout.addWidget(self.submit_button)

        self.save_button = QPushButton("导出识别结果")
        # self.save_button.clicked.connect(self.apply_data_range_to_view)
        parameter_settings_layout.addWidget(self.save_button)
        parameter_settings_layout.addStretch(1)

        left_splitter.addWidget(self.parameter_settings)

        left_splitter.setSizes([300, 500])  # Example sizes

        # 右侧的图展示区域
        right_section_widget = QWidget()
        right_section_layout = QVBoxLayout(right_section_widget)

        # 主图区
        self.main_plot = QWidget()
        self.main_plot.setObjectName("main_plot")
        main_plot_layout = QVBoxLayout(self.main_plot)
        # main_plot_layout.addWidget(QLabel("待载入数据"))

        # 添加 Matplotlib Canvas
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        main_plot_layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)

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
        plot1_container_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        # Plot 1 Navigation Buttons
        plot1_nav_layout = QHBoxLayout()
        plot1_nav_layout.addStretch()  # Push buttons to the right
        self.prev_plot1_button = QPushButton("<")
        self.next_plot1_button = QPushButton(">")

        self.prev_plot1_button.clicked.connect(self.show_prev_peak)
        self.next_plot1_button.clicked.connect(self.show_next_peak)

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
        # plot1_data_layout.addWidget(QLabel("图1的数据 Placeholder"))
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

        main_layout.addWidget(left_splitter)
        main_layout.addWidget(right_section_widget)

    @staticmethod
    # ====== 第一组：可选参数（带复选框）======
    def add_checkbox_spinbox(label_text, default_value=1.0, decimals=False, checked=True):
        container = QWidget()
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

        return container, checkbox, spinbox

    @staticmethod
    # ====== 第二组：颜色参数（无复选框）======
    def add_color_selector(default_color="#FF0000"):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel()
        label.setFixedWidth(60)
        label.setStyleSheet(f"background-color: {default_color}; border: 1px solid #666;")

        button = QPushButton("选择")

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

    def add_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "",
                                                   "数据文件 (*.csv *.txt *.abf);;所有文件 (*)")
        if file_path:
            self.data_manager.add_file(file_path)
            # self.file_list.addItem(file_path)
            self.refresh_list()

    def remove_file(self):
        item = self.file_list.currentItem()
        if item:
            row = self.file_list.row(item)

            if 0 <= row < len(self.data_file_paths):
                removed_path = self.data_file_paths.pop(row)
                self.data_manager.remove_file(removed_path)
                # print(self.data_manager.get_all_files())
                self.refresh_list()
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

    def refresh_list(self):
        self.trigger_refresh()
        # print("调用refresh")
        self.file_list.clear()
        self.data_file_paths.clear()
        # print(self.data_manager.get_all_files())
        for path in self.data_manager.get_all_files():
            self.data_file_paths.append(path)
            item = QListWidgetItem(os.path.basename(path))
            item.setToolTip(path)
            self.file_list.addItem(item)

    def update_load_time(self):
        # print("update_load_time 被调用了")
        if self.load_start_time is not None:
            elapsed = time.time() - self.load_start_time
            self.load_time_label.setText(f"加载时间：{elapsed:.2f} 秒")

    def load_selected_file(self, item):
        self.index = self.file_list.row(item)
        self.filepath = self.data_file_paths[self.index]
        print(self.filepath)
        self.load_data(self.filepath)

    def load_data(self, filepath):
        print("调用加载函数")
        if not filepath:
            return

        ext = os.path.splitext(filepath)[-1].lower()
        print(f"开始加载 {filepath}...")

        try:
            self.data = None
            self.full_x = None
            self.full_y = None
            self.data_length = 0
            self.peaks = []  # 存储峰值索引

            if ext == ".abf":
                if pyabf is None:
                    self.show_error("错误", "未安装 pyabf 库...")
                    return
                abf = pyabf.ABF(filepath)
                abf.setSweep(0)
                x = abf.sweepX  # 时间点
                y = abf.sweepY  # 个数

            elif ext in [".csv", ".txt"]:
                try:
                    print("使用 numpy.loadtxt 加载，大文件可能需要较长时间和较多内存...")
                    data_load = np.loadtxt(filepath, delimiter=',', comments='#')
                except ValueError:
                    data_load = np.loadtxt(filepath, comments='#')
                except MemoryError:
                    self.show_error("内存错误", "加载文件时内存不足，请尝试使用更小的数据集或优化加载方式。")
                    return

                if data_load.ndim == 1:
                    x = np.arange(len(data_load))
                    y = data_load
                elif data_load.shape[1] >= 2:
                    x = data_load[:, 0]
                    y = data_load[:, 1]
                    if data_load.shape[1] > 2:
                        print(f"警告: 文件包含超过2列，仅使用前两列。")
                else:
                    raise ValueError("数据格式无法解析为 X, Y 列")
            else:
                self.show_error("错误", f"不支持的文件类型: {ext}")
                return

            self.full_x = x  # 时间点
            self.full_y = y  # 数据点
            self.data_length = len(x)
            self.data = (x, y)

        except Exception as e:
            self.show_error("数据加载/处理失败", f"处理文件 '{os.path.basename(filepath)}' 时出错:\n{e}")

    def apply_data_range_to_view(self):
        if self.flag:
            self.flag = False
            self.load_start_time = time.time()
            self.load_timer.start(50)  # 每 50 毫秒刷新一次标签
            print("定时器启动了")

            # 创建线程和 worker
            self.thread = QThread()
            self.worker = LoadWorker(self)
            self.worker.moveToThread(self.thread)

            # 连接信号槽
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.on_loading_finished)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            # 启动线程
            self.thread.start()

    def on_loading_finished(self):
        self.load_timer.stop()
        self.flag = True

        self.peaks, self.prominences, self.height, self.width = self.worker.peaks  # 从 worker 拿回来

        self.plot_main()
        self.current_peak_index = 0
        self.plot_single_peak()

        final_time = time.time() - self.load_start_time
        self.load_time_label.setText(f"加载时间：{final_time:.2f} 秒")
        print(f"检测到 {len(self.peaks)} 个峰值")
        print(f"数据加载完成，耗时: {final_time:.2f} 秒")

    def data_peak(self):
        # 取反信号
        inverted_signal = -self.full_y
        # print(inverted_signal)
        # print(self.spin_height.logical_value)
        # print(self.spin_threshold.logical_value)
        # print(self.spin_width.logical_value)
        # print(self.spin_distance.logical_value)
        # print(self.spin_prominence.logical_value)
        peaks, properties = find_peaks(inverted_signal,
                                       height=self.spin_height.logical_value,
                                       threshold=self.spin_threshold.logical_value,
                                       distance=self.spin_distance.logical_value,
                                       prominence=self.spin_prominence.logical_value,
                                       width=self.spin_width.logical_value)
        prominences = peak_prominences(inverted_signal, peaks)[0]
        width_full = peak_widths(inverted_signal, peaks, rel_height=0.9)
        results_half = peak_widths(inverted_signal, peaks, rel_height=0.5)

        # 1.3 计算峰高 (负峰的绝对值)
        heights = np.abs(self.full_y[peaks])
        print(f"计算了 {len(heights)} 个峰的高度。")

        return peaks, prominences, heights, width_full

    def plot_main(self):
        if self.full_x is None or self.full_y is None:
            return

        self.ax.clear()

        self.ax.plot(self.full_x, self.full_y, label="Original Signal", color="blue")

        if self.peaks is not None:
            peak_times = self.full_x[self.peaks]
            peak_values = self.full_y[self.peaks]
            self.ax.plot(peak_times, peak_values, "ro", label="Peaks")

        self.ax.set_title("信号与峰值", fontproperties=get_chinese_font())
        # self.ax.figure.suptitle("主图区域", fontsize=14)
        self.ax.legend()
        self.canvas.draw()

    def plot_single_peak(self):
        # print("绘制单峰型图")
        if not hasattr(self, 'peaks') or len(self.peaks) == 0:
            return

        peak_idx = self.peaks[self.current_peak_index]
        # print(f"当前索引:{peak_idx}")
        window = 100  # 展示峰形前后点数
        start = max(0, peak_idx - window)
        end = min(len(self.full_y), peak_idx + window)

        x = self.full_x[start:end]
        y = self.full_y[start:end]
        local_peak = np.argmax(-y)
        inverted = -y

        # 半高宽计算
        self.results_full = peak_widths(inverted, [local_peak], rel_height=0.9)
        self.results_half = peak_widths(inverted, [local_peak], rel_height=0.5)

        # 删除旧图
        if hasattr(self, 'plot1_canvas'):
            self.plot1_layout.removeWidget(self.plot1_canvas)
            self.plot1_canvas.setParent(None)

        fig = Figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        ax.plot(x, y, label="Signal")
        ax.plot(x[local_peak], y[local_peak], "ro", label="Valley")

        # 半高宽线
        left_ips = int(self.results_full[2][0])
        right_ips = int(self.results_full[3][0])
        height = -self.results_full[1][0]
        ax.hlines(height, x[left_ips], x[right_ips], color="green", linewidth=2, label="Width")

        half_height = -self.results_half[1][0]
        ax.hlines(half_height, x[self.results_half[2][0].astype(int)], x[self.results_half[3][0].astype(int)],
                  color="yellow",
                  linewidth=2, label="Half_width")

        ax.set_title(f"Valley {self.current_peak_index + 1}/{len(self.peaks)}")
        ax.legend()
        ax.grid(True)

        self.plot1_canvas = FigureCanvas(fig)
        # self.plot1_layout = QVBoxLayout()
        self.plot1_canvas.setParent(self.plot1)
        self.plot1_layout.addWidget(self.plot1_canvas)
        self.plot1.setLayout(self.plot1_layout)
        self.plot1_canvas.draw()

        # 更新数据区显示
        for i in reversed(range(self.plot1_data_layout.count())):
            widget_to_remove = self.plot1_data_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

        label_texts = [
            f"index: {self.current_peak_index + 1} / {len(self.peaks)}",
            f"position (x，y):   ({x[local_peak]:.2f}，{y[local_peak]:.2f})",
            f"Left IP (x): {x[left_ips]:.2f}    ;    "
            f"Right IP (x): {x[right_ips]:.2f}",
            f"Width: {(x[right_ips] - x[left_ips]):.5f}",
            f"Height: {(height - y[local_peak]):.2f}",
            f"Half Height: {(half_height - y[local_peak]):.2f}"
        ]
        for text in label_texts:
            self.plot1_data_layout.addWidget(QLabel(text))

    def submit_data(self):
        if self.peaks is None or len(self.peaks) == 0:
            QMessageBox.warning(self, "提示", "当前没有可提交的识别结果。")
            return

        name, ok = QInputDialog.getText(self, "命名识别结果", "请输入该识别结果的名称：")
        if ok and name.strip():
            submission = {
                "name": name.strip(),
                "path": self.filepath,
                "peaks": self.peaks,
                "full_width": self.width,
                # "half_width": self.results_half,
                # "height": self.results_half,
                "prominences": self.prominences,
            }
            self.data_manager.add_peaks(submission)
            QMessageBox.information(self, "提交成功", f"识别结果已保存为“{name}”。")

        elif ok:
            QMessageBox.warning(self, "提示", "名称不能为空，请重新提交。")

    def show_prev_peak(self):
        if self.current_peak_index > 0:
            self.current_peak_index -= 1
            self.plot_single_peak()

    def show_next_peak(self):
        if self.current_peak_index < len(self.peaks) - 1:
            self.current_peak_index += 1
            self.plot_single_peak()

    # 分页信号传递
    def trigger_refresh(self):
        self.request_refresh.emit()

    def show_error(self, title, message):
        """显示一个简单的错误消息框"""
        print(f"--- {title} ---")
        print(message)
        print("-----------------")
