import os
import glob
import threading
from PyQt5.QtWidgets import QWidget, QListWidgetItem, QMessageBox, QFileDialog
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot

from train.UI import TrainingUI
from train.train_bp import train_bp_model
from train.train_snn import train_snn_model

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class WorkerSignals(QObject):
    """
    定义工作线程信号
    """
    log_updated = pyqtSignal(str)
    training_finished = pyqtSignal(object) # 传递结果字典
    training_error = pyqtSignal(str)

class TrainingPage(QWidget):
    """
    训练页面逻辑类
    """
    def __init__(self, data_manager):
        super().__init__()
        self.data_manager = data_manager # 虽然这里可能暂时用不到 data_manager，但保持接口一致
        self.ui = TrainingUI(self)
        
        # 信号连接
        self.ui.import_csv_btn.clicked.connect(self.import_csv_files)
        self.ui.remove_csv_btn.clicked.connect(self.remove_csv_files)
        self.ui.import_abf_btn.clicked.connect(self.import_abf_files)
        self.ui.remove_abf_btn.clicked.connect(self.remove_abf_files)
        
        self.ui.model_type_combo.currentIndexChanged.connect(self.on_model_type_changed)
        self.ui.train_btn.clicked.connect(self.start_training)
        self.ui.save_model_btn.clicked.connect(self.save_model)
        
        # 初始化状态
        self.current_training_result = None

    def import_csv_files(self):
        """导入 CSV 分类结果文件"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择分类结果文件 (CSV)", "", "CSV Files (*.csv);;All Files (*)")
        if not files:
            return
        for filepath in files:
            self.add_file_to_list(self.ui.csv_list, filepath)

    def import_abf_files(self):
        """导入 ABF 原始数据文件"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择原始数据文件 (ABF)", "", "ABF Files (*.abf);;All Files (*)")
        if not files:
            return
        for filepath in files:
            self.add_file_to_list(self.ui.abf_list, filepath)

    def add_file_to_list(self, list_widget, filepath):
        """添加文件到指定列表，避免重复"""
        # 检查是否已存在
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item.data(100) == filepath:
                return # 已存在

        item = QListWidgetItem(os.path.basename(filepath))
        item.setToolTip(filepath)
        item.setData(100, filepath) # 存储绝对路径
        list_widget.addItem(item)

    def remove_csv_files(self):
        """移除选中的 CSV 文件"""
        self.remove_selected_files(self.ui.csv_list)

    def remove_abf_files(self):
        """移除选中的 ABF 文件"""
        self.remove_selected_files(self.ui.abf_list)

    def remove_selected_files(self, list_widget):
        """从指定列表移除选中的文件"""
        selected_items = list_widget.selectedItems()
        if not selected_items:
            return
            
        for item in selected_items:
            row = list_widget.row(item)
            list_widget.takeItem(row)

    def on_model_type_changed(self, index):
        """模型类型切换时更新参数堆栈"""
        self.ui.params_stack.setCurrentIndex(index)

    def start_training(self):
        """开始训练"""
        # 1. 获取选中的文件 (CSV)
        # 这里逻辑修改：默认使用列表中的所有文件，或者只使用选中的文件？
        # 通常如果没有选中，就使用全部。如果有选中，就只使用选中。
        # 简单起见，这里我们获取列表中的所有文件。因为有两个列表，用户可能只想训练一部分，但多选比较麻烦。
        # 我们可以遵循这样的逻辑：
        # CSV列表：如果有选中项，则只处理选中的 CSV。否则处理所有 CSV。
        # ABF列表：作为资源池，全部传入。
        
        csv_items = self.ui.csv_list.selectedItems()
        if not csv_items:
            # 如果没选中，则取全部
            csv_items = [self.ui.csv_list.item(i) for i in range(self.ui.csv_list.count())]
            
        if not csv_items:
            QMessageBox.warning(self, "警告", "请至少导入一个分类结果文件 (CSV)！")
            return
            
        csv_files = [item.data(100) for item in csv_items]
        
        # 获取 ABF 文件池
        # 逻辑修改：如果有选中的 ABF 文件，只使用选中的。
        # 如果没有选中的，但列表中有文件，则全部使用？不，用户要求“要运行那个我鼠标选择的”
        # 所以：如果有选中，用选中。如果没有选中，但有文件，用全部？还是报错？
        # 既然用户说“要运行那个我鼠标选择的”，那隐含的意思是：
        # 1. 优先取选中的文件。
        # 2. 如果没选中，为了方便，取全部（比如只有一个文件时，用户可能懒得点选）。
        # 但最严格的解释是：只取选中的。不过为了用户体验，通常如果没选中，就默认为全部。
        # 结合上下文“如果UI界面有导入两个abf文件，要运行那个我鼠标选择的”，
        # 说明当存在多个文件时，必须依赖选择。
        
        abf_items = self.ui.abf_list.selectedItems()
        if not abf_items:
            # 如果没选中，且列表不为空，则使用全部
            abf_items = [self.ui.abf_list.item(i) for i in range(self.ui.abf_list.count())]
            
        abf_files = [item.data(100) for item in abf_items]
            
        if not abf_files:
             QMessageBox.warning(self, "警告", "请至少导入并选择一个原始文件 (ABF)！")
             return
        
        # 2. 获取参数
        model_type = self.ui.model_type_combo.currentText()
        params = {}
        
        if "BP" in model_type:
            # BP 参数
            try:
                hl_str = self.ui.bp_hidden_layers.text().replace("，", ",")
                hidden_layers = tuple(map(int, hl_str.split(",")))
            except:
                QMessageBox.warning(self, "参数错误", "隐藏层结构格式错误，应为如 '100, 50'")
                return
            
            params = {
                'type': 'BP',
                'hidden_layers': hidden_layers,
                'learning_rate': self.ui.bp_learning_rate.value(),
                'epochs': self.ui.bp_epochs.value(),
                'batch_size': self.ui.bp_batch_size.value()
            }
        else:
            # SNN 参数
            params = {
                'type': 'SNN',
                'time_steps': self.ui.snn_time_steps.value() if hasattr(self.ui, 'snn_time_steps') else 20,
                'tau': self.ui.snn_tau.value() if hasattr(self.ui, 'snn_tau') else 2.0,
                'learning_rate': self.ui.snn_learning_rate.value() if hasattr(self.ui, 'snn_learning_rate') else 1e-3,
                'epochs': self.ui.snn_epochs.value() if hasattr(self.ui, 'snn_epochs') else 100,
                'batch_size': self.ui.snn_batch_size.value() if hasattr(self.ui, 'snn_batch_size') else 32
            }
            
        # 3. 准备 UI 状态
        self.ui.train_btn.setEnabled(False)
        self.ui.save_model_btn.setEnabled(False)
        self.ui.log_output.clear()
        self.ui.clear_plots()
        self.ui.log_output.append(f"开始训练 {model_type}...")
        
        # 4. 启动线程
        self.signals = WorkerSignals()
        self.signals.log_updated.connect(self.append_log)
        self.signals.training_finished.connect(self.on_training_finished)
        self.signals.training_error.connect(self.on_training_error)
        
        # 传递 csv_files 和 abf_files
        thread = threading.Thread(target=self._training_worker, args=(csv_files, abf_files, params, self.signals))
        thread.daemon = True
        thread.start()

    def _training_worker(self, csv_files, abf_files, params, signals):
        """后台训练工作函数"""
        def callback(msg):
            signals.log_updated.emit(str(msg))
            
        try:
            # 更新 data_files 参数结构，或者让 train_bp_model 接受 abf_files
            # 这里我们约定：data_files 传 csv_files，额外传 abf_files
            # 为了保持接口简单，我们可以把 abf_files 塞进 params 或者修改 train_bp_model 签名
            # 最好是修改 train_bp_model 签名，但为了兼容性，我们先把 abf_files 放进 params 吧？
            # 不，还是显式传递比较好。这里我直接修改 train_bp_model 的调用。
            
            # 由于 train_bp_model 目前只接受 data_files, params, callback
            # 我们将 abf_files 放入 params 中传递，这是一种最小侵入的改法
            params['abf_files'] = abf_files
            
            if params['type'] == 'BP':
                result = train_bp_model(csv_files, params, callback)
            else:
                result = train_snn_model(csv_files, params, callback)
            
            signals.training_finished.emit(result)
        except Exception as e:
            import traceback
            signals.training_error.emit(str(e) + "\n" + traceback.format_exc())

    @pyqtSlot(str)
    def append_log(self, msg):
        self.ui.log_output.append(msg)
        # 滚动到底部
        scrollbar = self.ui.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @pyqtSlot(object)
    def on_training_finished(self, result):
        if result is None:
            self.on_training_error("训练失败，未返回有效结果。请检查日志中的警告信息（如是否找到了 ABF 文件）。")
            return

        self.current_training_result = result
        self.ui.train_btn.setEnabled(True)
        self.ui.save_model_btn.setEnabled(True)
        self.ui.log_output.append("训练完成！")
        
        # 清空旧图
        self.ui.clear_plots()
        
        # 绘制 Loss 曲线
        if 'loss_curve' in result and result['loss_curve']:
            self.plot_loss(result['loss_curve'])
            
        # 绘制 Accuracy 曲线
        if 'accuracy_curve' in result and result['accuracy_curve']:
            self.plot_accuracy(result['accuracy_curve'])
            
        # 显示分类报告
        if 'report' in result:
            self.ui.log_output.append("\n=== Classification Report ===\n")
            self.ui.log_output.append(str(result['report']))

    @pyqtSlot(str)
    def on_training_error(self, err_msg):
        self.ui.train_btn.setEnabled(True)
        self.ui.log_output.append(f"Error: {err_msg}")
        QMessageBox.critical(self, "训练出错", f"训练过程中发生错误：\n{err_msg}")

    def plot_loss(self, loss_data):
        """绘制 Loss 曲线"""
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(loss_data, label='Training Loss', color='blue')
        ax.set_title("Training Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        self.ui.add_plot(fig)

    def plot_accuracy(self, acc_data):
        """绘制 Accuracy 曲线"""
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(acc_data, label='Training Accuracy', color='green')
        ax.set_title("Training Accuracy")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        self.ui.add_plot(fig)

    def save_model(self):
        """保存模型"""
        if not self.current_training_result:
            return
            
        # 默认文件名
        default_name = "bp_model.pkl" if "classes" in self.current_training_result else "snn_model.pkl"
        
        file_path, _ = QFileDialog.getSaveFileName(self, "保存模型", default_name, "Pickle Files (*.pkl);;All Files (*)")
        
        if file_path:
            import joblib
            try:
                # 确保保存的数据包含模型名称 (用于 DataManager 识别)
                save_data = self.current_training_result.copy()
                if 'name' not in save_data:
                    save_data['name'] = os.path.splitext(os.path.basename(file_path))[0]
                
                joblib.dump(save_data, file_path)
                self.ui.log_output.append(f"模型已保存至: {file_path}")
                QMessageBox.information(self, "成功", "模型保存成功！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存模型失败: {str(e)}")
