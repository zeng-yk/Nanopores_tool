# load_worker.py
from PyQt5.QtCore import QObject, pyqtSignal

class LoadWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window  # 引用主界面

    def run(self):
        self.peaks = self.main_window.data_peak()
        self.finished.emit()