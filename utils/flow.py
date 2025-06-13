import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import Normalize, LinearSegmentedColormap

# 设置要可视化的 h5 文件路径
file_path = 'F:/data/data/drift_dataset/ntu_feature/out33_1.h5'  # 替换为实际路径
# 指定保存路径
output_dir = r'C:\Users\10508\Desktop\fighting\毕业\图\drift'
os.makedirs(output_dir, exist_ok=True)  # 如果目录不存在，则创建
# 打开 h5 文件
with h5py.File(file_path, 'r') as f:
    # 读取光流 x 和 y 数据集
    features_x = np.array(f['x'])
    features_y = np.array(f['y'])

# 要可视化的帧对索引
frame_pairs = [(7, 9), (9, 11), (11, 13), (13, 15), (15,17)]

# 创建彩虹渐变的颜色映射（从紫色到红色）
colors = [
    (0.5, 0, 0.5, 1),    # 紫色
    (0, 0, 1, 1),        # 蓝色
    (0, 1, 1, 1),        # 青色
    (0, 1, 0, 1),        # 绿色
    (1, 1, 0, 1),        # 黄色
    (1, 0.5, 0, 1),      # 橙色
    (1, 0, 0, 1)         # 红色
]
cmap = LinearSegmentedColormap.from_list("TransparentToRainbow", colors)

# 设置阈值，小于该阈值的区域将显示为透明
threshold = 1  # 可根据需要调整阈值

# 遍历每一对帧，计算光流差异并可视化
for (start, end) in frame_pairs:
    # 获取帧对的光流 x 和 y 分量
    flow_x_start = features_x[start]
    flow_y_start = features_y[start]
    flow_x_end = features_x[end]
    flow_y_end = features_y[end]

    # 计算帧对之间的光流差异
    flow_diff_x = flow_x_end - flow_x_start
    flow_diff_y = flow_y_end - flow_y_start

    # 计算光流差异的幅度
    magnitude_diff = np.sqrt(flow_diff_x**2 + flow_diff_y**2)

    # 设置透明度掩码
    alpha_mask = np.where(magnitude_diff > threshold, 1, 0)  # 设置透明度掩码

    # 设置小块大小以增加密集度
    block_size = 1
    magnitude_diff_resized = cv2.resize(magnitude_diff, (magnitude_diff.shape[1] // block_size, magnitude_diff.shape[0] // block_size), interpolation=cv2.INTER_AREA)
    magnitude_diff_resized = cv2.resize(magnitude_diff_resized, (magnitude_diff.shape[1], magnitude_diff.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 将颜色映射与透明度结合
    rgba_magnitude = cmap(Normalize(vmin=0, vmax=magnitude_diff_resized.max())(magnitude_diff_resized))
    rgba_magnitude[..., -1] = alpha_mask  # 应用透明度掩码

    # 绘制图像
    plt.figure(figsize=(8, 8))
    plt.imshow(rgba_magnitude)
    plt.title(f'Optical Flow Feature Visualization (Frames {start} to {end})')
    plt.axis('off')
    # 保存图像
    output_path = os.path.join(output_dir, f'optical_flow_{start}_{end}.png')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f'Saved optical flow visualization for frames {start} to {end} at {output_path}')
