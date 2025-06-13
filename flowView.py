import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 读取 HDF5 文件中的光流数据
def load_optical_flow_from_h5(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        # 假设光流数据分别保存在 'x' 和 'y' 数据集中，分别对应水平和垂直光流
        flow_x = f['x'][:]
        flow_y = f['y'][:]
    return flow_x, flow_y

# 可视化光流特征
def visualize_optical_flow(flow_x, flow_y, frame_idx):
    # 选择某一帧的光流
    fx = flow_x[frame_idx]
    fy = flow_y[frame_idx]

    # 计算光流的幅度和角度
    magnitude, angle = cv2.cartToPolar(fx, fy)

    # 通过幅度和角度构建彩色光流图
    hsv = np.zeros((fx.shape[0], fx.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # 角度映射到 [0, 180]
    hsv[..., 1] = 255  # 饱和度为 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # 幅度标准化到 [0, 255]

    # 转换为 BGR 格式以便显示
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 显示光流图像
    plt.imshow(flow_rgb)
    plt.title(f"Optical Flow for Frame {frame_idx}")
    plt.axis('off')
    plt.show()

# 示例文件路径
h5_file_path = 'F:/data/data/drift_dataset/ntu_feature/out33_1.h5'

# 读取光流数据
flow_x, flow_y = load_optical_flow_from_h5(h5_file_path)

# 可视化指定帧的光流（例如第0帧到第4帧）
for frame_idx in range(5):
    visualize_optical_flow(flow_x, flow_y, frame_idx)
