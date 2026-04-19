#!/usr/bin/env python3
"""
修复matplotlib字体警告问题
解决'Font default does not have a glyph for \u2212'警告
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

# 完全抑制字体警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*font.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*glyph.*')

# 设置matplotlib配置，彻底解决减号字符问题
plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial'],
    'axes.unicode_minus': False,  # 使用ASCII减号
    'text.usetex': False,  # 禁用LaTeX
    'axes.formatter.use_mathtext': False,  # 禁用数学文本格式
    'axes.formatter.useoffset': False,
    'axes.formatter.limits': [-5, 5],
    'font.family': 'sans-serif',
})

# 测试配置
print("Matplotlib字体配置:")
print(f"  font.sans-serif: {plt.rcParams['font.sans-serif']}")
print(f"  axes.unicode_minus: {plt.rcParams['axes.unicode_minus']}")
print(f"  text.usetex: {plt.rcParams['text.usetex']}")
print(f"  font.family: {plt.rcParams['font.family']}")

# 测试绘图
import numpy as np
fig, ax = plt.subplots()
x = np.array([1, 2, 3])
y = np.array([-1, -2, -3])
ax.plot(x, y, label='测试线')
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.legend()

# 检查警告
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    plt.savefig('test_font_fix.png', dpi=100)
    if w:
        print(f"仍有警告: {len(w)}个")
        for warning in w[:3]:
            print(f"  {warning.message}")
    else:
        print("无字体警告")

plt.close()

print("\n字体配置已更新，建议在所有绘图脚本开头导入此配置:")
print("from fix_matplotlib_font import plt  # 使用修复后的plt")
print("或直接在脚本开头添加:")
print("import matplotlib")
print("matplotlib.use('Agg')")
print("import matplotlib.pyplot as plt")
print("plt.rcParams.update({...})")