import sys
import os
import numpy as np
import time

from PyQt5.QtWidgets import QMainWindow, QListWidgetItem, QFileDialog
from PyQt5.QtCore import Qt, pyqtSignal
import pyqtgraph as pg

from .UI import DataViewerUI

try:
    import pyabf
except ImportError:
    pyabf = None

class DataViewer(QMainWindow):
    """数据查看器的主要类，处理业务逻辑"""
    def __init__(self, data_manager):
        super().__init__()

        self.data_manager = data_manager
        self.pages = {}

        # 数据相关属性
        self.data_file_paths = []  # 存放路径
        self.index = 0
        self.data = None # 存储完整的 (x, y) 数据
        self.full_x = None # 单独存储完整的 x 数据，方便查找索引
        self.full_y = None # 单独存储完整的 y 数据
        self.main_curve = None # 主图当前显示的曲线对象
        self.thumbnail_curve = None # 缩略图的曲线对象
        self.updating_range_programmatically = False # 用于防止信号循环触发的标志
        self.data_length = 0

        # 颜色设置
        self.color_item_pen = '#000000'
        self.color_line = '#0000ff'

        # 初始化UI
        self.ui = DataViewerUI(self)

        # 连接信号槽
        self.connect_signals()

        # 刷新文件列表
        self.refresh_list()

    def connect_signals(self):
        """连接所有的信号槽"""
        # 文件操作信号连接
        self.ui.import_button.clicked.connect(self.import_data_file)
        self.ui.delete_button.clicked.connect(self.delete_selected_file)
        self.ui.file_list.itemClicked.connect(self.load_selected_file)

        # 颜色设置信号连接
        self.ui.color_line_combo.lineEdit().textChanged.connect(self.update_color_line_preview)

        # 区域选择信号连接
        self.ui.region.sigRegionChanged.connect(self.on_region_changed)
        self.ui.main_plot_widget.sigXRangeChanged.connect(self.on_main_xrange_changed)
        self.ui.main_plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)

        # 应用范围按钮
        self.ui.apply_range_button.clicked.connect(self.apply_data_range_to_view)

        # 定时器信号连接
        self.ui.update_timer.timeout.connect(self.actually_update_main_plot)

        # 页面切换信号连接
        self.ui.button_view_data.clicked.connect(self.show_data_view_page)
        self.ui.button_view_analysis.clicked.connect(self.show_analysis_page)
        self.ui.button_view_clustering.clicked.connect(self.show_clustering_page)
        self.ui.button_view_train.clicked.connect(self.show_train_page)
        self.ui.button_view_predict.clicked.connect(self.show_predict_page)
        self.ui.button_view_settings.clicked.connect(self.show_settings_page)

        # 统计分析页面信号
        self.ui.btn_refresh_stats.clicked.connect(self.calculate_statistics)
        self.ui.page_tabs.currentChanged.connect(self.on_tab_changed)

    def refresh_list(self):
        """刷新文件列表"""
        self.ui.file_list.clear()
        self.data_file_paths.clear()
        for path in self.data_manager.get_all_files():
            self.data_file_paths.append(path)
            item = QListWidgetItem(os.path.basename(path))
            item.setToolTip(path)
            self.ui.file_list.addItem(item)

    def import_data_file(self):
        """导入数据文件"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择数据文件", "", "ABF 文件 (*.abf)")
        if not files:
            return

        for i, filepath in enumerate(files):
            if filepath not in self.data_file_paths:
                self.data_manager.add_file(filepath)
                self.data_file_paths.append(filepath)
                self.refresh_list()

                # 如果是第一个文件，立即加载
                if len(self.data_file_paths) == 1 and i == 0:
                    self.load_data(filepath)
                print(len(self.data_file_paths))

                # 如果有分析页面，刷新其列表
                if 'analysis' in self.pages:
                    self.pages['analysis'].refresh_list()

    def delete_selected_file(self):
        """删除选中的文件"""
        current_item = self.ui.file_list.currentItem()
        if current_item:
            row = self.ui.file_list.row(current_item)

            if 0 <= row < len(self.data_file_paths):
                removed_path = self.data_file_paths.pop(row)
                self.data_manager.remove_file(removed_path)
                print(self.data_manager.get_all_files())
                self.refresh_list()

                # 如果有分析页面，刷新其列表
                if 'analysis' in self.pages:
                    self.pages['analysis'].refresh_list()

                print(f"已移除路径: {removed_path}")

            # 强制取消选中项，避免误触发
            self.ui.file_list.clearSelection()

            # 若当前文件正是被删除的那个，且列表已空，清空数据
            if len(self.data_file_paths) == 0:
                self.clear_plots()
                self.ui.label_count.setText("未加载")
                self.data = None
                self.full_x = None
                self.full_y = None
                self.data_length = 0
            elif self.data_file_paths:
                self.load_data(self.data_file_paths[0])

    def load_selected_file(self, item):
        """加载用户选中的文件"""
        self.index = self.ui.file_list.row(item)
        print(self.ui.file_list.row(item))
        filepath = self.data_file_paths[self.index]
        print(filepath)
        self.load_data(filepath)

    def load_data(self, filepath):
        """加载数据文件"""
        print("调用加载函数")
        if filepath:
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
                    if pyabf is None:
                        self.show_error("错误", "未安装 pyabf 库...")
                        return
                    abf = pyabf.ABF(filepath)
                    abf.setSweep(0)
                    x = abf.sweepX
                    y = abf.sweepY
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
                self.ui.label_count.setText(f"{self.data_length:,}") # 使用千位分隔符
                self.ui.spin_start.setMaximum(self.data_length - 1 if self.data_length > 0 else 0)
                self.ui.spin_end.setMaximum(self.data_length - 1 if self.data_length > 0 else 0)
                self.ui.spin_start.setValue(0)
                self.ui.spin_end.setValue(self.data_length - 1 if self.data_length > 0 else 0)

                # 设置绘图
                self.setup_plots_after_load()

            except Exception as e:
                self.show_error("数据加载/处理失败", f"处理文件 '{os.path.basename(filepath)}' 时出错:\n{e}")
                self.clear_plots()

    def update_color_line_preview(self):
        """更新线条颜色预览"""
        print("调用数据点颜色变换")
        color = self.ui.color_line_combo.currentText()
        self.ui.color_line_preview.setStyleSheet(f"background-color: {color}; border: 1px solid gray;")
        self.color_line = color  # 绑定到变量
        self.load_data(self.data_file_paths[self.index])

    def setup_plots_after_load(self):
        """数据加载完成后，设置缩略图和主图的初始状态"""
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
        self.ui.thumbnail_plot_widget.clear()
        self.ui.thumbnail_plot_widget.disableAutoRange()


        # --- 3. 执行峰值保持降采样并绘制 ---
        thumbnail_target_points = 1000 # 目标"区间"数量，绘制点数约为2倍
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

        self.thumbnail_curve = self.ui.thumbnail_plot_widget.plot(
            x_thumb_display, y_thumb_display,
            pen=pg.mkPen(color='gray', width=0.5)
        )

        # --- 4. 显式设置缩略图的范围 ---
        print(f"设置缩略图范围: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}] (Y轴边距 0.1)")
        self.ui.thumbnail_plot_widget.setXRange(x_min, x_max, padding=0)
        self.ui.thumbnail_plot_widget.setYRange(y_min, y_max, padding=0.1)

        # --- 5. 验证缩略图范围 ---
        vb_thumb = self.ui.thumbnail_plot_widget.getViewBox()
        current_x_range = vb_thumb.viewRange()[0]
        current_y_range = vb_thumb.viewRange()[1]
        print(f"缩略图实际视图范围: X={current_x_range}, Y={current_y_range}")

        # --- 6. 添加 LinearRegionItem ---
        self.ui.thumbnail_plot_widget.addItem(self.ui.region, ignoreBounds=True)

        # --- 7. 设置主图的初始视图 ---
        initial_view_ratio = 0.01
        initial_x_end = x_min + (x_max - x_min) * initial_view_ratio
        initial_x_end = min(initial_x_end, x_max)
        print(f"设置主图初始 X 轴范围: [{x_min}, {initial_x_end}]")
        self.updating_range_programmatically = True
        self.ui.region.setRegion([x_min, initial_x_end])
        self.ui.main_plot_widget.setYRange(y_min, y_max, padding=0.1)
        self.ui.main_plot_widget.setXRange(x_min, initial_x_end, padding=0)
        self.updating_range_programmatically = False

        setup_end_time = time.time()
        print(f"绘图设置完成，耗时: {setup_end_time - setup_start_time:.2f}秒")

    def on_region_changed(self):
        """当缩略图区域改变时调用 (用户拖动/调整区域)"""
        if self.updating_range_programmatically or self.full_x is None:
            return
        # 设置主图的X范围，这将触发 on_main_xrange_changed
        minX, maxX = self.ui.region.getRegion()
        self.ui.main_plot_widget.setXRange(minX, maxX, padding=0, update=False) # 暂时不触发更新
        # 使用定时器延迟更新，避免拖动时频繁绘制
        self.ui.update_timer.start()

    def on_main_xrange_changed(self):
        """当主图X范围改变时调用 (用户缩放/平移主图 或 程序设置范围)"""
        if self.full_x is None:
            return

        # 1. 更新缩略图的 Region
        if not self.updating_range_programmatically: # 只有当用户交互导致范围改变时才更新region
            self.updating_range_programmatically = True # 防止 setRegion 再次触发自己
            minX, maxX = self.ui.main_plot_widget.getViewBox().viewRange()[0]
            self.ui.region.setRegion([minX, maxX])
            self.updating_range_programmatically = False

        # 2. 使用定时器延迟更新主图数据，避免过于频繁的计算和绘制
        self.ui.update_timer.start()

    def actually_update_main_plot(self):
        """实际执行主图数据更新和重绘的函数 (使用峰值保持降采样)"""
        if self.full_x is None or self.data_length == 0:
            print("无数据，跳过主图更新")
            return

        # 获取当前主图的X轴可视范围
        x_min_view, x_max_view = self.ui.main_plot_widget.getViewBox().viewRange()[0]
        update_start_time = time.time()

        # --- 查找数据索引 ---
        start_index = np.searchsorted(self.full_x, x_min_view, side='left')
        end_index = np.searchsorted(self.full_x, x_max_view, side='right')
        start_index = max(0, start_index - 1)  # 包含左边界附近的一个点
        end_index = min(self.data_length, end_index + 1)  # 包含右边界附近的一个点

        if start_index >= end_index:
            print("警告：计算出的索引范围无效，清空主图")
            if self.main_curve:
                self.ui.main_plot_widget.removeItem(self.main_curve)
                self.main_curve = None
            return

        # 提取需要显示的数据子集
        x_subset = self.full_x[start_index:end_index]
        y_subset = self.full_y[start_index:end_index]
        points_in_subset = len(x_subset)

        # 降采样处理
        target_intervals_main = int(self.ui.main_plot_widget.width() * 1.5)
        target_intervals_main = max(target_intervals_main, 250)
        target_intervals_main = min(target_intervals_main, 25000)

        if points_in_subset > target_intervals_main * 2:
            # 计算降采样步长
            main_sampling_step = max(1, points_in_subset // target_intervals_main)
            # 计算实际能生成的区间数量
            num_actual_intervals = points_in_subset // main_sampling_step

            # 预分配数组
            x_display = np.zeros(num_actual_intervals * 2)
            y_display = np.zeros(num_actual_intervals * 2)
            valid_points_main = 0

            downsample_start_time = time.time()
            # 在 subset 上执行 min/max 采样
            for i in range(num_actual_intervals):
                # 这里的 start_sub, end_sub 是相对于 subset 的索引
                start_sub = i * main_sampling_step
                end_sub = min(start_sub + main_sampling_step, points_in_subset)
                if start_sub >= end_sub: continue

                # 获取当前区间的数据 (从 subset 中获取)
                y_sub_interval = y_subset[start_sub:end_sub]

                # 计算区间内的min和max
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
        else:
            # 如果子集中的点数不多，直接绘制所有点
            x_display = x_subset
            y_display = y_subset

        # 绘制
        if self.main_curve:
            # 优化：如果曲线对象已存在，尝试用 setData 更新
            try:
                self.main_curve.setData(x=x_display, y=y_display)
            except Exception as e:
                self.ui.main_plot_widget.removeItem(self.main_curve)
                self.main_curve = self.ui.main_plot_widget.plot(x_display, y_display, pen=self.color_line, name="Data")
        else:
            # 如果曲线不存在，则创建
            self.main_curve = self.ui.main_plot_widget.plot(x_display, y_display, pen=self.color_line, name="Data")

    def clear_plots(self):
        """清空主图和缩略图"""
        print("Clearing plots...")
        self.ui.main_plot_widget.clear()
        self.ui.thumbnail_plot_widget.clear()
        self.main_curve = None
        self.thumbnail_curve = None
        # 重新添加 region (如果它存在)
        if hasattr(self.ui, 'region'):
             self.ui.thumbnail_plot_widget.addItem(self.ui.region, ignoreBounds=True)
        self.ui.coord_label.setText("坐标: N/A")
        self.ui.label_count.setText("未加载")
        self.ui.spin_start.setValue(0)
        self.ui.spin_end.setValue(0)
        self.ui.spin_start.setMaximum(0)
        self.ui.spin_end.setMaximum(0)

    def on_mouse_moved(self, event):
        """处理主图上的鼠标移动事件，显示坐标"""
        if not self.ui.main_plot_widget.plotItem.vb.sceneBoundingRect().contains(event):
             return # 鼠标不在绘图区

        vb = self.ui.main_plot_widget.plotItem.vb
        mouse_point = vb.mapSceneToView(event)
        x_coord = mouse_point.x()
        y_coord = mouse_point.y()
        self.ui.coord_label.setText(f"坐标: X={x_coord:.4f}, Y={y_coord:.4f}")

    def apply_data_range_to_view(self):
        """应用起始索引和结束索引设置到主图的视图范围"""
        if self.full_x is not None and self.data_length > 0:
            start_idx = self.ui.spin_start.value()
            end_idx = self.ui.spin_end.value()

            # 索引验证
            if start_idx >= self.data_length: start_idx = self.data_length - 1
            if end_idx >= self.data_length: end_idx = self.data_length - 1
            if start_idx < 0: start_idx = 0
            if end_idx < start_idx: end_idx = start_idx

            # 更新SpinBox的值为有效值
            self.ui.spin_start.setValue(start_idx)
            self.ui.spin_end.setValue(end_idx)

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
            self.ui.main_plot_widget.setXRange(start_x, end_x, padding=0)
            # 自动调整Y轴 (可选，但推荐)
            y_subset = self.full_y[start_idx : end_idx + 1]
            if len(y_subset) > 0:
                 self.ui.main_plot_widget.setYRange(np.min(y_subset), np.max(y_subset), padding=0.1)

            # 更新缩略图区域
            self.ui.region.setRegion([start_x, end_x])
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

    # --- 页面切换的槽函数 ---
    def show_data_view_page(self):
        self.ui.main_stack.setCurrentIndex(0)
        print("切换到 数据视图 页面")

    def show_analysis_page(self):
        # 从新的模块路径导入 AnalysisPage
        from analysis.analysis import AnalysisPage
        if 'analysis' not in self.pages:
            self.pages['analysis'] = AnalysisPage(self.data_manager)
            self.pages['analysis'].request_refresh.connect(self.refresh_list)
            self.ui.main_stack.addWidget(self.pages['analysis'])
        self.ui.main_stack.setCurrentWidget(self.pages['analysis'])
        print("切换到 分析 页面")

    def show_clustering_page(self):
        from clustering.clustering import ClusteringPage
        if 'clustering' not in self.pages:
            self.pages['clustering'] = ClusteringPage(self.data_manager)
            self.ui.main_stack.addWidget(self.pages['clustering'])
        self.ui.main_stack.setCurrentWidget(self.pages['clustering'])
        print("切换到 聚类 页面")

    def show_train_page(self):
        from train.train import TrainingPage
        if 'train' not in self.pages:
            self.pages['train'] = TrainingPage(self.data_manager)
            self.ui.main_stack.addWidget(self.pages['train'])
        self.ui.main_stack.setCurrentWidget(self.pages['train'])
        print("切换到 训练 页面")

    def show_predict_page(self):
        from predict.predict import PredictPage
        if 'predict' not in self.pages:
            self.pages['predict'] = PredictPage(self.data_manager)
            self.ui.main_stack.addWidget(self.pages['predict'])
        
        # 每次切换到预测页面时，强制刷新模型列表
        self.data_manager.load_models_from_disk()
        
        self.ui.main_stack.setCurrentWidget(self.pages['predict'])
        print("切换到 预测 页面")

    def show_settings_page(self):
        from settings.settings import SettingsPage
        if 'settings' not in self.pages:
            self.pages['settings'] = SettingsPage(self.data_manager)
            self.ui.main_stack.addWidget(self.pages['settings'])
        self.ui.main_stack.setCurrentWidget(self.pages['settings'])
        print("切换到 设置 页面")

    def on_tab_changed(self, index):
        """当标签页切换时调用"""
        # 如果切换到统计页 (假设索引为 1)，自动计算一次
        if index == 1 and self.full_x is not None:
            self.calculate_statistics()

    def calculate_statistics(self):
        """计算当前视图范围内的统计数据并绘制直方图"""
        if self.full_x is None or self.data_length == 0:
             # 避免未加载数据时切换标签页报错
             self.ui.lbl_mean.setText("均值: N/A")
             self.ui.lbl_std.setText("标准差: N/A")
             self.ui.lbl_min.setText("最小值: N/A")
             self.ui.lbl_max.setText("最大值: N/A")
             self.ui.histogram_plot.clear()
             return

        print("正在计算统计数据...")
        # 1. 获取当前视图范围
        vb = self.ui.main_plot_widget.getViewBox()
        x_min_view, x_max_view = vb.viewRange()[0]
        
        # 2. 查找数据索引
        start_index = np.searchsorted(self.full_x, x_min_view, side='left')
        end_index = np.searchsorted(self.full_x, x_max_view, side='right')
        
        # 边界保护
        start_index = max(0, start_index)
        end_index = min(self.data_length, end_index)
        
        if start_index >= end_index:
            print("当前视图范围内无数据点。")
            return

        # 3. 提取数据
        y_subset = self.full_y[start_index:end_index]
        
        if len(y_subset) == 0:
            return
            
        # 4. 计算统计量
        try:
            mean_val = np.mean(y_subset)
            std_val = np.std(y_subset)
            min_val = np.min(y_subset)
            max_val = np.max(y_subset)
            
            # 更新 UI
            self.ui.lbl_mean.setText(f"均值: {mean_val:.4f}")
            self.ui.lbl_std.setText(f"标准差: {std_val:.4f}")
            self.ui.lbl_min.setText(f"最小值: {min_val:.4f}")
            self.ui.lbl_max.setText(f"最大值: {max_val:.4f}")
            
            # 5. 计算和绘制直方图
            self.ui.histogram_plot.clear()
            
            # 自动决定 bin 的数量
            bins = 100
            if len(y_subset) > 10000:
                bins = 200
                
            y, x = np.histogram(y_subset, bins=bins)
            
            # 绘制 BarGraphItem
            # x 是 bin edges，长度比 y 多 1，取 bin 中心点
            x_centers = (x[:-1] + x[1:]) / 2
            width = (x[1] - x[0]) * 0.9 # 稍微留点缝隙
            
            bar_item = pg.BarGraphItem(x=x_centers, height=y, width=width, brush='#6666ff')
            self.ui.histogram_plot.addItem(bar_item)
            
            # 自动调整直方图视图
            self.ui.histogram_plot.autoRange()
            
            print("统计计算完成。")
            
        except Exception as e:
            print(f"统计计算出错: {e}")
