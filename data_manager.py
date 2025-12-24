# data_manager.py
import os
import pickle
import joblib
from PyQt5.QtCore import pyqtSignal, QObject


class DataManager(QObject):
    submissions_changed_signal = pyqtSignal()
    models_changed_signal = pyqtSignal()  # 新增：模型列表变化信号

    def __init__(self):
        super().__init__()
        self.submissions = []
        self.data_file_paths = []

        self.trained_models = []  # 新增：存储训练模型的列表
        
        # 模型保存路径
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        # 启动时加载模型
        self.load_models_from_disk()

    def load_models_from_disk(self):
        """从磁盘加载模型"""
        print("DataManager: 正在从磁盘加载模型...")
        self.trained_models = []
        if os.path.exists(self.model_dir):
            for filename in os.listdir(self.model_dir):
                if filename.endswith('.pkl'):
                    try:
                        path = os.path.join(self.model_dir, filename)
                        
                        # 优先使用 joblib 加载 (兼容性更好)
                        try:
                            model_info = joblib.load(path)
                        except Exception:
                            # 如果 joblib 失败，回退到 pickle
                            with open(path, 'rb') as f:
                                model_info = pickle.load(f)

                        # 简单验证一下是不是我们需要的格式
                        if isinstance(model_info, dict):
                            # 兼容性修复: 如果没有 name 字段，自动使用文件名
                            if 'name' not in model_info:
                                model_info['name'] = os.path.splitext(filename)[0]
                            
                            self.trained_models.append(model_info)
                            print(f"DataManager: 已加载模型 '{model_info['name']}'")
                    except Exception as e:
                        print(f"DataManager: 加载模型文件 {filename} 失败: {e}")
        
        self.models_changed_signal.emit()


    def add_file(self, path):
        print("传入文件" + format(path))
        if path not in self.data_file_paths:
            self.data_file_paths.append(path)

    def remove_file(self, path):
        print("移除文件" + format(path))
        if path in self.data_file_paths:
            self.data_file_paths.remove(path)

    def get_all_files(self):
        return self.data_file_paths

    def add_peaks(self, submission: dict):
        print(f"成功传入" + format(submission))
        self.submissions.append(submission)
        self.submissions_changed_signal.emit()  # <--- 在数据变化后发射信号

    def get_submission_names(self):
        """
        从 submissions 列表中提取所有字典的 'name' 键的值。
        :return: 一个包含所有 'name' 值的字符串列表。
                 如果字典没有 'name' 键，则使用 '未命名' 代替。
        """
        # 使用列表推导式遍历 self.submissions 列表
        # 对于列表中的每个字典 (sub)，使用 sub.get('name', '未命名') 获取 'name' 的值
        # .get() 方法比直接用 sub['name'] 更安全，因为它允许指定一个默认值 ('未命名')
        # 防止字典没有 'name' 键，避免程序出错。
        names_list = [sub.get('name', '未命名') for sub in self.submissions]
        print(f"DataManager: 提取到的名称列表: {names_list}")  # 调试信息
        return names_list

    def get_data_by_name(self, name):
        """
        根据 submission 的名称查找并返回用于聚类的数据。
        !!! 你需要根据你的实际数据结构来实现这个方法 !!!
        """
        print(f"DataManager: 尝试查找名为 '{name}' 的数据...")
        for submission in self.submissions:
            if submission.get('name') == name:
                # 假设你的 submission 字典中有一个键（例如 'processed_data' 或 'features'）
                # 存储着可以直接用于聚类的数值型数据 (如 numpy array)
                path = submission.get('path')
                peaks = submission.get('peaks')  # <-- 数据键名
                full_width = submission.get('full_width')
                # half_width = submission.get('half_width')
                prominences = submission.get('prominences')
#                 height = submission.get('height')
                if peaks is not None:
                    print(f"DataManager: 找到数据，类型: {type(peaks)}")
                    # return path, peaks, full_width, half_width, prominences,height
                    return path, peaks, full_width, prominences
                else:
                    print(f"DataManager: 找到名为 '{name}' 的 submission，但缺少 'data' 键或其值为 None。")
                    return None
        print(f"DataManager: 未找到名为 '{name}' 的 submission。")
        return None  # 循环结束还没找到，返回 None

    # --- 新增模型管理方法 ---

    def add_model(self, model_info: dict):
        """保存模型信息"""
        self.trained_models.append(model_info)
        print(f"DataManager: 模型 '{model_info.get('name')}' 已保存。")
        self.models_changed_signal.emit()

    def get_model_names(self):
        """获取所有模型名称"""
        return [m.get('name', '未命名模型') for m in self.trained_models]

    def get_model_by_name(self, name):
        """获取模型对象和标签映射"""
        for m in self.trained_models:
            if m.get('name') == name:
                return m
        return None
