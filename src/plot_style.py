"""共享绘图样式：注册 Windows 中文字体，统一报告图表风格。
所有生成报告图表的脚本都应 import 本模块并调用 setup_chinese_font()。
"""
import matplotlib
from matplotlib import font_manager


def setup_chinese_font():
    """注册中文字体并设为默认，返回实际使用的字体名。"""
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",    # 黑体
        "C:/Windows/Fonts/simsun.ttc",    # 宋体
    ]
    chosen = None
    for path in candidates:
        try:
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            chosen = name
            break
        except Exception:
            continue
    if chosen:
        matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False  # 负号正常显示
    return chosen
