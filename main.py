"""
Nanopores tool - 应用程序启动入口
"""
import sys
import os
from PyQt5.QtWidgets import QApplication
from data_manager import DataManager
from home.home import DataViewer

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 初始化数据管理器
    data_manager = DataManager()

    # 实例化并显示主界面
    viewer = DataViewer(data_manager)
    viewer.show()

    # 启动应用
    sys.exit(app.exec_())
