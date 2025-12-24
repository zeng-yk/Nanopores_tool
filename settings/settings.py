import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QTextEdit, QFileDialog, 
    QFrame, QGroupBox, QSpacerItem, QSizePolicy, QMessageBox
)
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QPixmap, QDesktopServices, QFont

class SettingsPage(QWidget):
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Top Header Section: Logo + Title + Github
        header_layout = QHBoxLayout()
        
        # 1. School Logo (Top Left)
        logo_label = QLabel()
        logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'media', '校徽.jpeg')
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            # Scale it down if it's too big, e.g., height 80
            pixmap = pixmap.scaledToHeight(80, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        else:
            logo_label.setText("Logo Image Not Found")
            logo_label.setStyleSheet("border: 1px solid gray; padding: 10px;")
        
        header_layout.addWidget(logo_label)
        
        # Title and Info
        title_layout = QVBoxLayout()
        title_label = QLabel("Nanopores Analysis Tool")
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        title_layout.addWidget(title_label)
        
        # 2. Github Link
        self.github_url = "https://github.com/your-username/nanopores-tool" # Placeholder
        github_label = QLabel(f'<a href="{self.github_url}">访问 GitHub 项目主页</a>')
        github_label.setOpenExternalLinks(True) # Allow clicking to open browser
        github_label.setFont(QFont("Arial", 12))
        title_layout.addWidget(github_label)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch() # Push everything to left
        
        main_layout.addLayout(header_layout)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)

        # 3. Model Path Configuration
        model_group = QGroupBox("模型路径设置")
        model_layout = QHBoxLayout()
        
        self.path_edit = QLineEdit()
        self.path_edit.setText(self.data_manager.model_dir)
        self.path_edit.setReadOnly(True) # Read-only, use button to change
        model_layout.addWidget(self.path_edit)
        
        browse_btn = QPushButton("更改路径")
        browse_btn.clicked.connect(self.change_model_path)
        model_layout.addWidget(browse_btn)
        
        reset_btn = QPushButton("重置默认")
        reset_btn.clicked.connect(self.reset_model_path)
        model_layout.addWidget(reset_btn)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # 4. Open Source License
        license_group = QGroupBox("开源许可证 (MIT License)")
        license_layout = QVBoxLayout()
        
        license_text = QTextEdit()
        license_text.setReadOnly(True)
        license_text.setText(self.get_license_text())
        license_layout.addWidget(license_text)
        
        license_group.setLayout(license_layout)
        main_layout.addWidget(license_group)

    def change_model_path(self):
        directory = QFileDialog.getExistingDirectory(self, "选择模型保存路径", self.data_manager.model_dir)
        if directory:
            self.path_edit.setText(directory)
            self.data_manager.model_dir = directory
            # Reload models from the new directory
            try:
                self.data_manager.load_models_from_disk()
                QMessageBox.information(self, "成功", f"模型路径已更新为:\n{directory}\n\n已重新加载该目录下的模型。")
            except Exception as e:
                QMessageBox.warning(self, "警告", f"加载新目录下的模型时出错:\n{str(e)}")

    def reset_model_path(self):
        # Default path: project_root/model
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
        self.path_edit.setText(default_path)
        self.data_manager.model_dir = default_path
        self.data_manager.load_models_from_disk()
        QMessageBox.information(self, "已重置", "模型路径已恢复默认。")

    def get_license_text(self):
        return """MIT License

Copyright (c) 2024 Nanopores Analysis Tool Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
