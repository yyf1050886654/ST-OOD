import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['<Microsoft YaHei>']
plt.rcParams['font.sans-serif'] = ['<Microsoft YaHei>']
plt.rcParams['xtick.labelsize'] = 13  # 设置x轴刻度标签的字体大小
plt.rcParams['ytick.labelsize'] = 13  # 设置y轴刻度标签的字体大小
The_measured_FDR = [0.025, 0.026, 0.027, 0.028, 0.028, 0.03, 0.03, 0.03, 0.031]
# The_measured_FDR=[0.036, 0.037, 0.038,0.04, 0.041,0.042, 0.042,0.042,0.043]
The_theoretical_FDR_upper_bound = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
x = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
plt.rcParams.update({'figure.figsize': [8, 5]})
ax = plt.gca()
plt.xticks(x, x)

# 设置标签的字体大小
plt.plot(x, The_measured_FDR, label=r"$\mathbf{The\ measured\ FDR}$", marker="o", color='red', clip_on=False,linewidth=3)
plt.plot(x, The_theoretical_FDR_upper_bound, label=r"$\mathbf{The\ theoretical\ FDR\ upper\ bound}$", marker="+", color='g', clip_on=False)
# plt.plot(x, The_measured_FDR, label="The measured FDR", marker="o", color='g', clip_on=False)
# plt.plot(x, The_theoretical_FDR_upper_bound, label="The theoretical FDR upper bound", marker="+", color='purple', clip_on=False)

# 设置图例字体大小
legend = ax.legend()
for text in legend.get_texts():
    text.set_fontsize(13)  # 设置标签的字体大小
    # text.set_color('b')

# 设置最黑的网格线颜色
plt.grid(True, linestyle="--", alpha=0.8, color='black')

# 设置轴边框线的样式
# for spine in ax.spines.values():
#     spine.set_linewidth(2)  # 设置边框线宽度

plt.xlim([0.1, 0.5])
plt.ylim([0.00, 0.5])
formula = r'$\frac{1}{\lambda} \left(\frac{c+1}{c}\right)^s$'
plt.subplots_adjust(bottom=0.15)
plt.xlabel(formula, fontsize=15, fontweight='bold')
plt.ylabel("FDR", fontsize=15)

# 设置y轴刻度
y_ticks = np.arange(0, 0.51, 0.05)
plt.yticks(y_ticks)

# plt.savefig('snowy_fdr.jpg')
plt.savefig("drift_fdr.eps", format="eps")
plt.show()
