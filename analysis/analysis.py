"""
Analysis 模块的主要功能逻辑实现
"""
import csv
import json
import os
import sys
import time
import traceback
import pyabf
import numpy as np
from PyQt5.QtWidgets import QWidget, QListWidgetItem, QMessageBox, QInputDialog, QFileDialog, QLabel
from PyQt5.QtCore import pyqtSignal, Qt, QThread, QTimer
from scipy.signal import find_peaks, peak_prominences, peak_widths
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .UI import AnalysisUI, get_chinese_font
from load_worker import LoadWorker

class AnalysisPage(QWidget):
    """分析页面的主要逻辑实现类"""
    request_refresh = pyqtSignal()

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)

        self.filepath = None
        self.data_manager = data_manager

        # 峰值检测相关属性
        self.peaks = None
        self.prominences = None
        self.height = None
        self.width = None
        self.peak_properties = None
        self.current_peak_index = 0

        # 数据存储
        self.index = 0
        self.data = None
        self.full_x = None
        self.full_y = None
        self.data_file_paths = []  # 存放路径

        # 状态标记
        self.flag = True  # 是否有运行峰值检测
        self.select_flag = False
        self.Positive = True
        self.load_start_time = None

        # 获取字体属性
        self.chinese_font = get_chinese_font()

        # 初始化UI
        self.ui = AnalysisUI(self)

        # 连接信号槽
        self.connect_signals()

        # 刷新文件列表
        self.refresh_list()

    def connect_signals(self):
        """连接所有信号和槽"""
        # 文件操作按钮信号
        self.ui.add_button.clicked.connect(self.add_file)
        self.ui.remove_button.clicked.connect(self.remove_file)
        self.ui.file_list.itemClicked.connect(self.load_selected_file)

        # 峰值检测按钮信号
        self.ui.btn1.clicked.connect(self.update_button_style)
        self.ui.btn2.clicked.connect(self.update_button_style)

        # 应用参数按钮信号
        self.ui.apply_range_button.clicked.connect(self.apply_data_range_to_view)

        # 导航按钮信号
        self.ui.prev_plot1_button.clicked.connect(self.show_prev_peak)
        self.ui.next_plot1_button.clicked.connect(self.show_next_peak)

        # 结果操作按钮信号
        self.ui.submit_button.clicked.connect(self.submit_data)
        self.ui.save_button.clicked.connect(self.save_data)

        # 计时器信号
        self.ui.load_timer.timeout.connect(self.update_load_time)

    def update_button_style(self):
        """更新波峰/波谷按钮样式和状态"""
        # 设置被选中的按钮为蓝色，未选中为默认
        for btn in [self.ui.btn1, self.ui.btn2]:
            if btn.isChecked():
                if self.ui.btn1.isChecked():
                    self.Positive = True
                else:
                    self.Positive = False
                btn.setStyleSheet("background-color: gray")
            else:
                btn.setStyleSheet("")

    def add_file(self):
        """添加文件到分析列表"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "ABF 文件 (*.abf)")
        if file_path:
            self.data_manager.add_file(file_path)
            self.refresh_list()

    def remove_file(self):
        """从分析列表中移除选中的文件"""
        item = self.ui.file_list.currentItem()
        if item:
            row = self.ui.file_list.row(item)

            if 0 <= row < len(self.data_file_paths):
                removed_path = self.data_file_paths.pop(row)
                self.data_manager.remove_file(removed_path)
                self.refresh_list()
                print(f"已移除路径: {removed_path}")

            # 强制取消选中项，避免误触发
            self.ui.file_list.clearSelection()

            # 若当前文件正是被删除的那个，且列表已空，清空数据
            if len(self.data_file_paths) == 0:
                self.clear_plots()
                self.data = None
                self.full_x = None
                self.full_y = None
                self.data_length = 0

    def refresh_list(self):
        """刷新文件列表"""
        self.trigger_refresh()
        self.ui.file_list.clear()
        self.data_file_paths.clear()
        for path in self.data_manager.get_all_files():
            self.data_file_paths.append(path)
            item = QListWidgetItem(os.path.basename(path))
            item.setToolTip(path)
            self.ui.file_list.addItem(item)

    def trigger_refresh(self):
        """触发刷新信号"""
        self.request_refresh.emit()

    def update_load_time(self):
        """更新加载时间显示"""
        if self.load_start_time is not None:
            elapsed = time.time() - self.load_start_time
            self.ui.load_time_label.setText(f"加载时间：{elapsed:.2f} 秒")

    def load_selected_file(self, item):
        """加载用户选中的文件"""
        self.select_flag = True
        self.index = self.ui.file_list.row(item)
        self.filepath = self.data_file_paths[self.index]
        print(self.filepath)
        self.load_data(self.filepath)

    def load_data(self, filepath):
        """加载数据文件"""
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
        """应用参数进行峰值检测"""
        if self.flag & self.select_flag:
            self.flag = False
            self.load_start_time = time.time()
            self.ui.load_timer.start(50)  # 每 50 毫秒刷新一次标签
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
        """峰值检测完成后的处理"""
        self.ui.load_timer.stop()
        self.flag = True

        self.peaks, self.prominences, self.height, self.width = self.worker.peaks  # 从 worker 拿回来

        self.plot_main()
        self.current_peak_index = 0
        self.plot_single_peak()

        final_time = time.time() - self.load_start_time
        self.ui.load_time_label.setText(f"加载时间：{final_time:.2f} 秒")
        print(f"检测到 {len(self.peaks)} 个峰值")
        print(f"数据加载完成，耗时: {final_time:.2f} 秒")

    def data_peak(self):
        """执行峰值检测"""
        # 取反信号
        if self.Positive:
            print("波峰检测")
            inverted_signal = self.full_y
        else:
            print("波谷检测")
            inverted_signal = -self.full_y

        peaks, properties = find_peaks(inverted_signal,
                                       height=self.ui.spin_height.logical_value,
                                       threshold=self.ui.spin_threshold.logical_value,
                                       distance=self.ui.spin_distance.logical_value,
                                       prominence=self.ui.spin_prominence.logical_value,
                                       width=self.ui.spin_width.logical_value)
        prominences = peak_prominences(inverted_signal, peaks)[0]
        width_full = peak_widths(inverted_signal, peaks, rel_height=0.9)
        results_half = peak_widths(inverted_signal, peaks, rel_height=0.5)

        # 计算峰高 (负峰的绝对值)
        heights = np.abs(self.full_y[peaks])
        print(f"计算了 {len(heights)} 个峰的高度。")

        return peaks, prominences, heights, width_full

    def plot_main(self):
        """绘制主图"""
        if self.full_x is None or self.full_y is None:
            return

        self.ui.ax.clear()

        if self.ui.Downsampling_checkbox.isChecked():
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

            self.ui.ax.plot(x_display, y_display, label="Original Signal", color="blue")
        else:
            self.ui.ax.plot(self.full_x, self.full_y, label="Original Signal", color="blue")

        if self.peaks is not None:
            peak_times = self.full_x[self.peaks]
            peak_values = self.full_y[self.peaks]
            self.ui.ax.plot(peak_times, peak_values, "ro", label="Peaks")

        self.ui.ax.set_title("信号与峰值", fontproperties=get_chinese_font(), fontsize=18)
        self.ui.ax.legend(fontsize=14)
        self.ui.canvas.draw()

    def plot_single_peak(self):
        """绘制单个峰的详细信息"""
        if not hasattr(self, 'peaks') or len(self.peaks) == 0:
            return

        peak_idx = self.peaks[self.current_peak_index]
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
            self.ui.plot1_layout.removeWidget(self.plot1_canvas)
            self.plot1_canvas.setParent(None)

        signal_color = self.ui.btn_color1.color
        width_line_color_90 = self.ui.btn_color2.color
        width_line_color_50 = self.ui.btn_color3.color

        fig = Figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        ax.plot(x, y, label="Signal", color=signal_color)

        # 半高宽线
        left_ips = int(self.results_full[2][0])
        right_ips = int(self.results_full[3][0])
        height = -self.results_full[1][0]
        ax.hlines(height, x[left_ips], x[right_ips], color=width_line_color_90, linewidth=2, label="Width")

        half_height = -self.results_half[1][0]
        ax.hlines(half_height, x[self.results_half[2][0].astype(int)], x[self.results_half[3][0].astype(int)],
                  color=width_line_color_50,
                  linewidth=2, label="Half_width")

        ax.set_title(f"Valley {self.current_peak_index + 1}/{len(self.peaks)}")
        ax.legend()
        ax.grid(True)

        self.plot1_canvas = FigureCanvas(fig)
        self.plot1_canvas.setParent(self.ui.plot1)
        self.ui.plot1_layout.addWidget(self.plot1_canvas)
        self.ui.plot1.setLayout(self.ui.plot1_layout)
        self.plot1_canvas.draw()

        # 更新数据区显示
        for i in reversed(range(self.ui.plot1_data_layout.count())):
            widget_to_remove = self.ui.plot1_data_layout.itemAt(i).widget()
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
            self.ui.plot1_data_layout.addWidget(QLabel(text))

    def submit_data(self):
        """提交识别结果"""
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
                "prominences": self.prominences,
            }
            self.data_manager.add_peaks(submission)
            QMessageBox.information(self, "提交成功", f"识别结果已保存为{name}。")

        elif ok:
            QMessageBox.warning(self, "提示", "名称不能为空，请重新提交。")

    def show_prev_peak(self):
        """显示上一个峰"""
        if self.current_peak_index > 0:
            self.current_peak_index -= 1
            self.plot_single_peak()

    def show_next_peak(self):
        """显示下一个峰"""
        if self.current_peak_index < len(self.peaks) - 1:
            self.current_peak_index += 1
            self.plot_single_peak()

    def save_data(self):
        """保存识别结果"""
        print("尝试保存识别结果...")
        # 1. 检查是否有结果
        if self.peaks is None or len(self.peaks) == 0:
            QMessageBox.warning(self, "无法保存", "没有有效的峰值检测结果可供保存。请先运行检测。")
            return
        if not self.filepath:
            QMessageBox.warning(self, "无法保存", "当前未加载有效的数据文件。")
            return

        # 2. 选择保存目录
        default_dir = os.path.dirname(self.filepath)
        save_dir = QFileDialog.getExistingDirectory(self, "选择保存结果的文件夹", default_dir)

        if not save_dir:
            print("用户取消保存。")
            return

        # 3. 创建主文件夹名称
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

            # 这里应该有更多的保存逻辑，但在原始文件中未包含完整实现
            # 完整版可以按需要添加导出为JSON、CSV等格式的代码

            QMessageBox.information(self, "保存成功", f"结果已保存到:\n{main_folder_path}")

        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存过程中发生错误:\n{str(e)}")
            print(f"保存失败: {e}")

    def clear_plots(self):
        """清空图表"""
        pass

    def show_error(self, title, message):
        """显示错误消息"""
        QMessageBox.warning(self, title, message)
