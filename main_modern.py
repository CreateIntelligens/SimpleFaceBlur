"""
人臉遮蔽工具 - 主程式（現代化版本）
Face Blur Tool - Main Entry Point (Modern UI)
"""

import sys
import os

# 確保能找到模組
if getattr(sys, 'frozen', False):
    # 如果是打包後的exe
    application_path = sys._MEIPASS
else:
    # 如果是原始碼執行
    application_path = os.path.dirname(os.path.abspath(__file__))

# 新增到系統路徑
sys.path.insert(0, application_path)

from gui_modern import main

if __name__ == "__main__":
    main()
