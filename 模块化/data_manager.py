# data_manager.py （也可以直接写在 MainWindow 类里）
class DataManager:
    def __init__(self):
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

    # def peaks(self):
    #