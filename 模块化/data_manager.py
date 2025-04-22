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
        # 以防某个字典碰巧没有 'name' 键，避免程序出错。
        names_list = [sub.get('name', '未命名') for sub in self.submissions]
        print(f"DataManager: 提取到的名称列表: {names_list}")  # 调试信息
        return names_list