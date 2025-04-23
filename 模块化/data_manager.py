# data_manager.py
from PyQt5.QtCore import pyqtSignal, QObject


class DataManager(QObject):
    submissions_changed_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.submissions = []
        self.data_file_paths = []

    def add_file(self, path):
        print("传入文件"+ format(path))
        if path not in self.data_file_paths:
            self.data_file_paths.append(path)

    def remove_file(self, path):
        print("移除文件"+ format(path))
        if path in self.data_file_paths:
            self.data_file_paths.remove(path)

    def get_all_files(self):
        return self.data_file_paths

    def add_peaks(self, submission: dict):
        print(f"成功传入"+ format(submission))
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
                data = submission.get('data')  # <-- 数据键名
                if data is not None:
                    print(f"DataManager: 找到数据，类型: {type(data)}")
                    return path,data
                else:
                    print(f"DataManager: 找到名为 '{name}' 的 submission，但缺少 'data' 键或其值为 None。")
                    return None
        print(f"DataManager: 未找到名为 '{name}' 的 submission。")
        return None  # 循环结束还没找到，返回 None