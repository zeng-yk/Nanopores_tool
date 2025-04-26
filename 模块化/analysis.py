# analysis_page.py
import csv
import json
import os
import time

import pyabf
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QListWidget, QPushButton, QFileDialog, QHBoxLayout, QListWidgetItem, \
    QSplitter, QFormLayout, QSpinBox, QColorDialog, QLabel, QDoubleSpinBox, QCheckBox, QMessageBox, QInputDialog, \
    QSizePolicy, QButtonGroup, QApplication, QProgressDialog
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from matplotlib import font_manager, pyplot as plt

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

        self.flag = True  # 是否有运行峰值检测
        self.select_flag = False
        self.Positive = True
        self.chinese_font = get_chinese_font() # 获取字体属性

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

        # 复选框
        self.Downsampling_checkbox = QCheckBox("性能模式")
        self.Downsampling_checkbox.setChecked(True)  # 默认勾选

        self.btn1 = QPushButton("波峰")
        self.btn2 = QPushButton("波谷")
        # 设置为可点击状态
        self.btn1.setCheckable(True)
        self.btn2.setCheckable(True)
        # 创建互斥按钮组
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)  # 设置互斥
        self.group.addButton(self.btn1)
        self.group.addButton(self.btn2)
        # 初始样式
        # self.update_button_style()
        # 连接信号
        self.btn1.clicked.connect(self.update_button_style)
        self.btn2.clicked.connect(self.update_button_style)
        Button_layout = QHBoxLayout()
        Button_layout.addWidget(self.btn1)
        Button_layout.addWidget(self.btn2)
        # self.setLayout(Button_layout)

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
        self.save_button.clicked.connect(self.save_data)
        parameter_settings_layout.addWidget(self.save_button)
        parameter_settings_layout.addStretch(1)

        left_splitter.addWidget(self.parameter_settings)

        left_splitter.setSizes([700, 300])  # Example sizes

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
    def add_label(name):
        label = QLabel(name)
        label.setFixedWidth(100)
        return label

    def update_button_style(self):
        # 设置被选中的按钮为蓝色，未选中为默认
        for btn in [self.btn1, self.btn2]:
            if btn.isChecked():
                if self.btn1.isChecked():
                    self.Positive = True
                else:
                    self.Positive = False
                btn.setStyleSheet("background-color: gray")
            else:
                btn.setStyleSheet("")

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
        self.select_flag = True
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
        if self.flag & self.select_flag:
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
        if self.Positive:
            print("波峰检测")
            inverted_signal = self.full_y
        else:
            print("波谷检测")
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

        if self.Downsampling_checkbox.isChecked():
            print("性能模式已开启")
            # --- 使用峰值保持降采样 ---
            thumbnail_target_points = 4000  # 采样点目标值
            data_length = len(self.full_y)
            thumbnail_sampling_step = max(1, data_length // thumbnail_target_points)
            num_intervals = data_length // thumbnail_sampling_step

            x_display = np.zeros(num_intervals * 2)
            y_display = np.zeros(num_intervals * 2)

            valid_points = 0
            for i in range(num_intervals):
                start = i * thumbnail_sampling_step
                end = min(start + thumbnail_sampling_step, data_length)
                if start >= end:
                    continue
                y_interval = self.full_y[start:end]
                interval_min = np.nanmin(y_interval)
                interval_max = np.nanmax(y_interval)

                x_coord = self.full_x[start]
                idx = valid_points * 2
                x_display[idx] = x_coord
                y_display[idx] = interval_min
                x_display[idx + 1] = x_coord
                y_display[idx + 1] = interval_max
                valid_points += 1

            x_display = x_display[:valid_points * 2]
            y_display = y_display[:valid_points * 2]

            self.ax.plot(x_display, y_display, label="Original Signal", color="blue")
        else:
            self.ax.plot(self.full_x, self.full_y, label="Original Signal", color="blue")

        if self.peaks is not None:
            peak_times = self.full_x[self.peaks]
            peak_values = self.full_y[self.peaks]
            self.ax.plot(peak_times, peak_values, "ro", label="Peaks")

        self.ax.set_title("信号与峰值", fontproperties=get_chinese_font())
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

    # --- save_data (核心实现) ---
    def save_data(self):
        print("尝试保存识别结果...")
        # 1. 检查是否有结果
        if self.peaks is None or len(self.peaks) == 0:
            QMessageBox.warning(self, "无法保存", "没有有效的峰值检测结果可供保存。请先运行检测。")
            return
        if not self.filepath:
            QMessageBox.warning(self, "无法保存", "当前未加载有效的数据文件。")
            return

        # 2. 选择保存目录
        # 建议默认保存目录为原始文件所在目录
        default_dir = os.path.dirname(self.filepath)
        save_dir = QFileDialog.getExistingDirectory(self, "选择保存结果的文件夹", default_dir)

        if not save_dir:
            print("用户取消保存。")
            return

        # 3. 创建主文件夹名称 (基于原始文件名和时间戳，避免覆盖)
        base_filename = os.path.splitext(os.path.basename(self.filepath))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        main_folder_name = f"{base_filename}_results_{timestamp}"
        main_folder_path = os.path.join(save_dir, main_folder_name)

        all_json_data = []
        all_csv_data_rows = []

        try:
            os.makedirs(main_folder_path, exist_ok=True)  # 创建主文件夹
            print(f"结果将保存到: {main_folder_path}")

            # 4. 创建子文件夹
            main_plot_dir = os.path.join(main_folder_path, "main_plot")
            single_plots_dir = os.path.join(main_folder_path, "single_peak_plots")
            single_data_dir = os.path.join(main_folder_path, "single_peak_data")
            os.makedirs(main_plot_dir, exist_ok=True)
            os.makedirs(single_plots_dir, exist_ok=True)
            os.makedirs(single_data_dir, exist_ok=True)

            # 5. 保存主图 (使用当前的 self.figure)
            main_plot_filename = os.path.join(main_plot_dir, f"{base_filename}_main_plot.png")
            try:
                self.figure.savefig(main_plot_filename, dpi=300, bbox_inches='tight')
                print(f"主图已保存到: {main_plot_filename}")
            except Exception as e:
                print(f"保存主图失败: {e}")
                QMessageBox.warning(self, "保存错误", f"保存主图时出错:\n{e}")
                # 不中断，继续尝试保存其他内容

            # 6. 循环处理并保存每个单峰图和数据
            num_peaks_to_save = len(self.peaks)
            progress = QProgressDialog(f"正在导出 {num_peaks_to_save} 个峰...", "取消", 0, num_peaks_to_save, self)
            progress.setWindowTitle("导出进度")
            progress.setWindowModality(Qt.WindowModal)  # 模态对话框，阻止其他操作
            progress.setMinimumDuration(1000)  # 1秒后才显示，避免闪烁
            progress.setValue(0)
            QApplication.processEvents()  # 显示进度条

            saved_count = 0
            for i in range(num_peaks_to_save):
                if progress.wasCanceled():
                    print("导出被用户取消。")
                    break

                progress.setValue(i)
                progress.setLabelText(f"正在导出峰 {i + 1}/{num_peaks_to_save}...")
                QApplication.processEvents()

                peak_global_idx = self.peaks[i]
                # --- 6a. 生成并保存单峰图 ---
                # (与 plot_single_peak 类似，但操作在临时的 Figure 上)
                try:
                    temp_fig, temp_ax = plt.subplots(figsize=(4, 3))  # 创建临时图
                    # (复用 plot_single_peak 的窗口计算和绘图逻辑，但绘制到 temp_ax 上)
                    # 这里简化，直接调用一个辅助函数或复制代码片段
                    plot_success, extracted_data = self._generate_single_peak_figure_and_data(i, temp_ax)

                    if plot_success:
                        peak_filename_base = f"peak_{i + 1:04d}"  # 格式化文件名，如 peak_0001
                        plot_filepath = os.path.join(single_plots_dir, f"{peak_filename_base}.png")
                        temp_fig.savefig(plot_filepath, dpi=150, bbox_inches='tight')
                        saved_count += 1
                    else:
                        print(f"警告：未能成功生成峰 {i + 1} 的图形。")

                    plt.close(temp_fig)  # **非常重要：关闭临时图形释放内存**

                except Exception as e_plot:
                    print(f"导出峰 {i + 1} 的图形时出错: {e_plot}")
                    plt.close(temp_fig)  # 确保关闭
                    extracted_data = None  # 图形失败，数据也可能不完整

                # --- 6b. 提取并保存单峰数据 (CSV 和 JSON) ---
                if extracted_data:  # 确保从绘图辅助函数获取了数据
                    # csv_filepath = os.path.join(single_data_dir, f"{peak_filename_base}_data.csv")
                    # json_filepath = os.path.join(single_data_dir, f"{peak_filename_base}_data.json")

                    # 准备 JSON 数据 (主要是汇总信息)
                    json_data = {
                        "peak_index_global": int(peak_global_idx),
                        "peak_index_in_list": i + 1,
                        "peak_x": extracted_data.get("peak_x"),
                        "peak_y": extracted_data.get("peak_y"),
                        "prominence": extracted_data.get("prominence"),
                        # 添加宽度信息
                        "width_90": extracted_data.get("width_90"),
                        "width_90_y_level": extracted_data.get("width_90_y_level"),
                        "width_90_left_x": extracted_data.get("width_90_left_x"),
                        "width_90_right_x": extracted_data.get("width_90_right_x"),
                        "width_50": extracted_data.get("width_50"),
                        "width_50_y_level": extracted_data.get("width_50_y_level"),
                        "width_50_left_x": extracted_data.get("width_50_left_x"),
                        "width_50_right_x": extracted_data.get("width_50_right_x"),
                        # 可以添加窗口范围信息
                        "window_start_index": extracted_data.get("window_start_index"),
                        "window_end_index": extracted_data.get("window_end_index"),
                    }
                    all_json_data.append(json_data)
                    # CSV 每行数据，保持和 JSON 一致
                    csv_row = [
                        json_data["peak_index_global"],
                        json_data["peak_index_in_list"],
                        json_data["peak_x"],
                        json_data["peak_y"],
                        json_data["prominence"],
                        json_data["width_90"],
                        json_data["width_90_y_level"],
                        json_data["width_90_left_x"],
                        json_data["width_90_right_x"],
                        json_data["width_50"],
                        json_data["width_50_y_level"],
                        json_data["width_50_left_x"],
                        json_data["width_50_right_x"],
                        json_data["window_start_index"],
                        json_data["window_end_index"],
                    ]
                    all_csv_data_rows.append(csv_row)
                else:
                    print(f"跳过保存峰 {i + 1} 的数据，因为绘图或数据提取失败。")


                #     # 保存 JSON
                #     try:
                #         with open(json_filepath, 'w', encoding='utf-8') as f_json:
                #             # 使用 numpy_encoder 处理可能存在的 numpy 类型
                #             json.dump(json_data, f_json, indent=4, default=self.numpy_encoder)
                #     except Exception as e_json:
                #         print(f"保存峰 {i + 1} 的 JSON 数据时出错: {e_json}")
                #
                #     # 准备并保存 CSV 数据 (主要是窗口内的 X, Y 数据)
                #     try:
                #         with open(csv_filepath, 'w', newline='', encoding='utf-8') as f_csv:
                #             writer = csv.writer(f_csv)
                #             # 写入头信息 (从 JSON 数据提取)
                #             for key, value in json_data.items():
                #                 writer.writerow([f"# {key}", value])
                #             writer.writerow([])  # 空行分隔
                #             writer.writerow(["X", "Y"])  # 数据列标题
                #             # 写入窗口数据
                #             x_win = extracted_data.get("x_window", [])
                #             y_win = extracted_data.get("y_window", [])
                #             if len(x_win) == len(y_win):
                #                 for row_idx in range(len(x_win)):
                #                     writer.writerow([x_win[row_idx], y_win[row_idx]])
                #             else:
                #                 print(f"警告：峰 {i + 1} 的窗口数据 X/Y 长度不匹配，CSV 未写入数据点。")
                #     except Exception as e_csv:
                #         print(f"保存峰 {i + 1} 的 CSV 数据时出错: {e_csv}")
                #
                #     saved_count += 1
                # else:
                #     print(f"跳过保存峰 {i + 1} 的数据，因为绘图或数据提取失败。")

            # 保存总的 JSON 文件
            all_json_path = os.path.join(single_data_dir, "all_peaks_data.json")
            try:
                with open(all_json_path, 'w', encoding='utf-8') as f_all_json:
                    json.dump(all_json_data, f_all_json, indent=4, default=self.numpy_encoder)
                print(f"所有峰的数据已保存到: {all_json_path}")
            except Exception as e_json_all:
                print(f"保存总 JSON 文件时出错: {e_json_all}")

            # 保存总的 CSV 文件
            all_csv_path = os.path.join(single_data_dir, "all_peaks_data.csv")
            try:
                with open(all_csv_path, 'w', newline='', encoding='utf-8') as f_all_csv:
                    writer = csv.writer(f_all_csv)
                    # 写入表头
                    writer.writerow([
                        "peak_index_global",
                        "peak_index_in_list",
                        "peak_x",
                        "peak_y",
                        "prominence",
                        "width_90",
                        "width_90_y_level",
                        "width_90_left_x",
                        "width_90_right_x",
                        "width_50",
                        "width_50_y_level",
                        "width_50_left_x",
                        "width_50_right_x",
                        "window_start_index",
                        "window_end_index",
                    ])
                    writer.writerows(all_csv_data_rows)
                print(f"所有峰的数据已保存到: {all_csv_path}")
            except Exception as e_csv_all:
                print(f"保存总 CSV 文件时出错: {e_csv_all}")

            progress.setValue(num_peaks_to_save)  # 完成进度条
            QApplication.processEvents()

            # 7. 显示最终结果
            if not progress.wasCanceled():
                QMessageBox.information(self, "导出完成",
                                        f"成功导出 {saved_count}/{num_peaks_to_save} 个峰的结果到:\n{main_folder_path}")
            else:
                QMessageBox.warning(self, "导出已取消",
                                    f"导出过程被取消。\n已导出 {saved_count} 个峰的结果到:\n{main_folder_path}")

        except OSError as e_os:
            print(f"创建文件夹结构时出错: {e_os}")
            QMessageBox.critical(self, "保存失败", f"无法创建保存目录结构:\n{e_os}")
        except Exception as e_main:
            print(f"保存过程中发生意外错误: {e_main}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "保存失败", f"保存过程中发生未知错误:\n{e_main}")

    def _generate_single_peak_figure_and_data(self, peak_list_index, ax):
        """
        辅助函数：为指定索引的峰生成图形（在传入的 Axes 上）并提取相关数据。
        **采用局部窗口计算宽度的方式，与 plot_single_peak 保持一致。**
        Args:
            peak_list_index (int): 峰在 self.peaks 列表中的索引。
            ax (matplotlib.axes.Axes): 要绘制图形的 Axes 对象。
        Returns:
            tuple: (bool, dict)
                   bool: 是否成功生成图形和提取数据。
                   dict: 包含该峰详细信息的字典，用于保存。
        """
        if self.peaks is None or not (0 <= peak_list_index < len(self.peaks)):
            print(f"Error (helper): Invalid peak_list_index: {peak_list_index}")
            return False, {}

        peak_global_idx = self.peaks[peak_list_index]
        if not (0 <= peak_global_idx < self.data_length):
            print(f"Error (helper): Invalid global index {peak_global_idx} for list index {peak_list_index}")
            return False, {}

        # --- 确定窗口 (使用与 plot_single_peak 相似的固定窗口或之前的动态逻辑) ---
        # 为了严格匹配 plot_single_peak 的示例，我们使用固定窗口
        # 如果需要动态窗口，可以取消注释下面的 width_data_90 部分
        window_half_width = 100  # 与 plot_single_peak 示例中的 window = 100 对应 (半宽)
        start_idx = max(0, int(peak_global_idx - window_half_width))
        end_idx = min(self.data_length, int(peak_global_idx + window_half_width + 1))  # +1 for slicing end point
        x_window = self.full_x[start_idx:end_idx]
        y_window = self.full_y[start_idx:end_idx]

        if len(x_window) == 0:
            print(f"Error (helper): Empty window data for peak {peak_list_index}")
            return False, {}

        # --- 定位峰在窗口内的索引 ---
        peak_local_idx = peak_global_idx - start_idx
        # 验证并重新定位 (如果计算出的索引无效或想更精确)
        if not (0 <= peak_local_idx < len(y_window)):
            print(f"Warning (helper): Calculated local index {peak_local_idx} out of bounds. Re-finding in window.")
            if self.Positive:  # Find maximum for peaks
                peak_local_idx = np.argmax(y_window)
            else:  # Find minimum for valleys
                peak_local_idx = np.argmin(y_window)
            # Final check
            if not (0 <= peak_local_idx < len(y_window)):
                print(f"Error (helper): Could not reliably find peak within the window for peak {peak_list_index}.")
                return False, {}

        peak_x_val = x_window[peak_local_idx]
        peak_y_val = y_window[peak_local_idx]

        # --- 准备用于局部宽度计算的信号 ---
        inverted_window_y = -y_window if not self.Positive else y_window

        # --- 在局部窗口内计算宽度 ---
        results_full_local = None
        results_half_local = None
        try:
            # Note: peak_widths needs peak indices as a list/array
            results_full_local = peak_widths(inverted_window_y, [peak_local_idx], rel_height=0.9)
        except ValueError as e:
            print(f"Helper: Warning - Could not calculate 90% width for peak {peak_list_index} in window: {e}")
        except Exception as e:  # Catch other potential errors
            print(f"Helper: Warning - Unexpected error calculating 90% width for peak {peak_list_index}: {e}")

        try:
            results_half_local = peak_widths(inverted_window_y, [peak_local_idx], rel_height=0.5)
        except ValueError as e:
            print(f"Helper: Warning - Could not calculate 50% width for peak {peak_list_index} in window: {e}")
        except Exception as e:
            print(f"Helper: Warning - Unexpected error calculating 50% width for peak {peak_list_index}: {e}")

        # --- 绘图 ---
        signal_color = self.btn_color1.color
        peak_color = "#FF0000"
        width_line_color_90 = self.btn_color2.color  # Use color 3 for 90%
        width_line_color_50 = self.btn_color3.color  # Example: Gold/Yellow for 50% (like plot_single_peak)

        peak_label_str = "波峰" if self.Positive else "波谷"
        if not self.chinese_font: peak_label_str = "Peak" if self.Positive else "Valley"

        ax.plot(x_window, y_window, color=signal_color, linewidth=1.5,
                label="信号段" if self.chinese_font else "Signal Segment")
        ax.plot(peak_x_val, peak_y_val, "o", markersize=6, color=peak_color, label=peak_label_str)

        # --- 准备要返回的数据字典 ---
        extracted_data = {
            "peak_index_global": int(peak_global_idx),
            "peak_index_in_list": peak_list_index + 1,
            "peak_x": peak_x_val,
            "peak_y": peak_y_val,
            "prominence": None,  # Will be filled later from self.prominences
            "width_90": None, "width_90_y_level": None, "width_90_left_x": None, "width_90_right_x": None,
            "width_50": None, "width_50_y_level": None, "width_50_left_x": None, "width_50_right_x": None,
            "x_window": x_window.tolist(),
            "y_window": y_window.tolist(),
            "window_start_index": start_idx,
            "window_end_index": end_idx,
        }

        # --- 绘制宽度线 (使用局部计算结果) ---
        if results_full_local and results_full_local[0] is not None and len(results_full_local[0]) > 0:
            try:
                # Indices are relative to the window start (start_idx)
                left_ips_local = int(np.floor(results_full_local[2][0]))  # Floor for safety indexing x_window
                right_ips_local = int(np.ceil(results_full_local[3][0]))  # Ceil for safety indexing x_window
                # Ensure indices are within the bounds of x_window
                left_ips_local = max(0, left_ips_local)
                right_ips_local = min(len(x_window) - 1, right_ips_local)

                height_level_local = results_full_local[1][0]  # Height level from peak_widths result

                # Get corresponding X coordinates from the window's X data
                left_x = x_window[left_ips_local]
                right_x = x_window[right_ips_local]

                # Determine the actual Y level on the plot
                # Height returned by peak_widths is relative to the baseline of the signal fed to it (inverted_window_y)
                actual_y_level = -height_level_local if not self.Positive else height_level_local

                label_90 = "90%宽度  " if self.chinese_font else "Width @ 90%"
                ax.hlines(actual_y_level, left_x, right_x, color=width_line_color_90, linestyle='--', linewidth=2,
                          label=label_90)
                ax.plot([left_x, right_x], [actual_y_level, actual_y_level], '|', color=width_line_color_90,
                        markersize=10)

                # Populate extracted_data
                extracted_data["width_90"] = right_x - left_x
                extracted_data["width_90_y_level"] = actual_y_level
                extracted_data["width_90_left_x"] = left_x
                extracted_data["width_90_right_x"] = right_x

            except IndexError as e:
                print(f"Helper: Error processing 90% width indices for peak {peak_list_index}: {e}")
            except Exception as e:
                print(f"Helper: Error plotting 90% width for peak {peak_list_index}: {e}")

        # 50% Width (matches 'results_half' in plot_single_peak)
        if results_half_local and results_half_local[0] is not None and len(results_half_local[0]) > 0:
            try:
                # Indices are relative to the window start (start_idx)
                left_ips_local = int(np.floor(results_half_local[2][0]))
                right_ips_local = int(np.ceil(results_half_local[3][0]))
                left_ips_local = max(0, left_ips_local)
                right_ips_local = min(len(x_window) - 1, right_ips_local)

                height_level_local = results_half_local[1][0]

                left_x = x_window[left_ips_local]
                right_x = x_window[right_ips_local]
                actual_y_level = -height_level_local if not self.Positive else height_level_local

                label_50 = "宽度 @ 50%" if self.chinese_font else "Width @ 50%"
                ax.hlines(actual_y_level, left_x, right_x, color=width_line_color_50, linestyle='-', linewidth=2,
                          label=label_50)
                ax.plot([left_x, right_x], [actual_y_level, actual_y_level], '|', color=width_line_color_50,
                        markersize=10)

                extracted_data["width_50"] = right_x - left_x
                extracted_data["width_50_y_level"] = actual_y_level
                extracted_data["width_50_left_x"] = left_x
                extracted_data["width_50_right_x"] = right_x

            except IndexError as e:
                print(f"Helper: Error processing 50% width indices for peak {peak_list_index}: {e}")
            except Exception as e:
                print(f"Helper: Error plotting 50% width for peak {peak_list_index}: {e}")

        # 提取突出度
        if hasattr(self, 'prominences') and self.prominences is not None and len(self.prominences) > peak_list_index:
            try:
                extracted_data["prominence"] = self.prominences[peak_list_index]
            except IndexError:
                print(f"Helper: Warning - Index out of bounds for prominence at peak {peak_list_index}")

        # --- 设置图形属性 ---
        title_text = f"{peak_label_str} {peak_list_index + 1} (全局索引 {peak_global_idx})"
        if not self.chinese_font: title_text = f"{peak_label_str} {peak_list_index + 1} (Global Index {peak_global_idx})"
        font_prop = self.chinese_font if self.chinese_font else None
        ax.set_title(title_text, fontproperties=font_prop, fontsize=9)
        ax.set_xlabel("时间/索引" if self.chinese_font else "Time/Index", fontproperties=font_prop)
        ax.set_ylabel("幅值" if self.chinese_font else "Amplitude", fontproperties=font_prop)
        # Only add legend if there are labeled elements
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles=handles, labels=labels, prop=font_prop if font_prop else None, fontsize=7)
        ax.grid(True, linestyle=':', alpha=0.7)

        return True, extracted_data

    @staticmethod
    def numpy_encoder(obj):
        """ 自定义 JSON 编码器，处理 numpy 类型 """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # 检查是否为 NaN 或 Inf
            if np.isnan(obj): return "NaN" # 将 NaN 转为字符串
            if np.isinf(obj): return "Infinity" if obj > 0 else "-Infinity" # 将 Inf 转为字符串
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist() # 将数组转为列表
        elif isinstance(obj, (np.bool_, bool)):
             return bool(obj)
        # 添加对 bytes 的处理 (如果需要)
        # elif isinstance(obj, bytes):
        #     try:
        #         return obj.decode('utf-8') # 尝试解码
        #     except UnicodeDecodeError:
        #         return f"bytes:{obj.hex()}" # 无法解码则返回十六进制表示
        try:
            # 尝试默认的 JSON 序列化
            return json.JSONEncoder().default(obj)
        except TypeError:
             # 如果还是失败，返回对象的字符串表示
             return str(obj)

    # 分页信号传递
    def trigger_refresh(self):
        self.request_refresh.emit()

    def show_error(self, title, message):
        """显示一个简单的错误消息框"""
        print(f"--- {title} ---")
        print(message)
        print("-----------------")
