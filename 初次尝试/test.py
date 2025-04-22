import os
from matplotlib import font_manager

for font in font_manager.findSystemFonts(fontpaths=["/System/Library/Fonts"]):
    # print(font)
    if "PingFang" in font or "Heiti" in font:
        print(font)