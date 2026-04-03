import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from tqdm import tqdm

from io_utils import get_image_files, read_image_grayscale
from kernels import get_kernel


class KDEBackgroundDetector:
    """
    基于核密度估计的运动目标检测器（灰度图像）
    """

    def __init__(self, bandwidth=30, threshold=1e-6, kernel='epanechnikov'):
        """
        初始化检测器
        :param bandwidth: 核带宽，控制对灰度值变化的敏感度（像素值范围 0-255）
        :param threshold: 前景判定阈值，低于此概率值判定为前景
        :param kernel: 核函数名称，支持 'epanechnikov' 或 'gaussian'
        """
        self.bandwidth = bandwidth
        self.threshold = threshold
        self.kernel_func = get_kernel(kernel)
        self.background_model = None  # shape: (h, w, n_samples)
        self.trained = False
        self.h = None
        self.w = None
        self.n_samples = None

    def train(self, train_frames_path, frame_extensions=None):
        """
        使用训练集图像训练背景模型
        :param train_frames_path: 训练图像文件夹路径
        :param frame_extensions: 支持的图片扩展名列表，默认 .jpg/.png/.jpeg
        :return: 是否训练成功
        """
        if not os.path.isdir(train_frames_path):
            print(f"训练文件夹不存在：{train_frames_path}")
            return False

        train_images = get_image_files(train_frames_path, frame_extensions)
        if len(train_images) == 0:
            print(f"错误：在 {train_frames_path} 中没有找到图像文件")
            return False
        print(f"找到 {len(train_images)} 张训练图像")

        # 读取第一张获取尺寸
        first_img = read_image_grayscale(train_images[0])
        if first_img is None:
            print(f"错误：无法读取图像 {train_images[0]}")
            return False

        h, w = first_img.shape
        n_samples = len(train_images)
        self.background_model = np.zeros((h, w, n_samples), dtype=np.uint8)

        print("正在加载训练图像...")
        for i, img_path in enumerate(tqdm(train_images)):
            img = read_image_grayscale(img_path)
            if img is not None:
                self.background_model[:, :, i] = img
            else:
                print(f"警告：无法读取图像 {img_path}，跳过")

        self.trained = True
        self.h, self.w = h, w
        self.n_samples = n_samples
        print(f"背景模型训练完成！模型尺寸: {h}x{w}, 每个像素有 {n_samples} 个历史样本")
        return True

    def detect_single_frame_vectorized(self, frame):
        """
        向量化检测单帧（输入可为灰度或彩色图）
        :param frame: 输入图像（numpy数组）
        :return: 前景掩码（0-255，前景为255）
        """
        if not self.trained:
            raise RuntimeError("必须先训练模型！")

        # 转换为灰度图
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # 尺寸对齐
        if gray.shape[0] != self.h or gray.shape[1] != self.w:
            gray = cv2.resize(gray, (self.w, self.h))

        # 扩展维度以广播：(h, w, 1) 与 (h, w, n_samples) 计算
        current = np.expand_dims(gray, axis=2)
        u = np.abs(current - self.background_model) / self.bandwidth

        # 调用核函数计算
        kernel_vals = self.kernel_func(u)
        # 密度估计 = (1/(N * bandwidth)) * ΣK(u)
        probabilities = np.sum(kernel_vals, axis=2) / (self.n_samples * self.bandwidth)

        mask_ = (probabilities < self.threshold).astype(np.uint8) * 255
        return mask_

    # ========== 最终版：三维KDE绘图（支持Z轴压平） ==========
    def plot_3d_kde(self, frame, step=5, elev=25, azim=60, cmap='inferno',
                    title=None, save_path=None, show=False, use_log_scale=True,
                    plot_type='surface', surface_quality='high', z_compression=0.3):
        """
        绘制高质量三维概率密度图（支持Z轴视觉压缩）
        :param frame: 输入图像（灰度或彩色）
        :param step: 采样步长（值越小越精细，但速度慢）
        :param elev: 仰角
        :param azim: 方位角
        :param cmap: 颜色映射（推荐 'inferno', 'plasma', 'viridis'）
        :param title: 图标题
        :param save_path: 保存路径
        :param show: 是否显示
        :param use_log_scale: 是否对密度取对数（强烈推荐 True）
        :param plot_type: 'surface' 或 'scatter'
        :param surface_quality: 'high' (全分辨率) 或 'low' (加快速度)
        :param z_compression: Z轴视觉压缩因子，小于1使图形更扁平（推荐 0.2~0.5）
        """
        if not self.trained:
            raise RuntimeError("必须先训练模型！")

        # 灰度转换与尺寸对齐
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        if gray.shape[0] != self.h or gray.shape[1] != self.w:
            gray = cv2.resize(gray, (self.w, self.h))

        # 计算概率密度矩阵
        current = np.expand_dims(gray, axis=2)
        u = np.abs(current - self.background_model) / self.bandwidth
        kernel_vals = self.kernel_func(u)
        probabilities = np.sum(kernel_vals, axis=2) / (self.n_samples * self.bandwidth)

        # 采样网格
        x_inds = np.arange(0, self.w, step)
        y_inds = np.arange(0, self.h, step)
        Z = probabilities[y_inds[:, None], x_inds]
        X, Y = np.meshgrid(x_inds, y_inds)

        # 对数缩放
        if use_log_scale:
            z_disp = np.log10(Z + 1e-12)
            zlabel = 'Log$_{10}$(Probability Density)'
        else:
            z_disp = Z
            zlabel = 'Probability Density'

        # 创建图形
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # 绘图
        if plot_type == 'scatter':
            sc = ax.scatter(X.ravel(), Y.ravel(), z_disp.ravel(),
                            c=z_disp.ravel(), cmap=cmap, s=5, alpha=0.8)
            cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
            cbar.set_label(zlabel, fontsize=10)
        else:  # surface
            if surface_quality == 'high':
                rstride, cstride = 1, 1
            else:
                rstride, cstride = max(1, step // 2), max(1, step // 2)
            surf = ax.plot_surface(X, Y, z_disp, cmap=cmap, edgecolor='none',
                                   rstride=rstride, cstride=cstride,
                                   alpha=0.9, antialiased=True)
            cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
            cbar.set_label(zlabel, fontsize=10)

        # 设置Z轴刻度为科学计数法
        if use_log_scale:
            ax.zaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.zaxis.get_major_formatter().set_powerlimits((0, 0))
            ax.zaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            ax.zaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis='z', style='sci', scilimits=(0, 0))

        # 关键：压平Z轴（设置三个轴的长度比例，Z轴比例 = z_compression）
        ax.set_box_aspect([1, 1, z_compression])

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"三维KDE图已保存至：{save_path}")
        if show:
            plt.show()
        plt.close(fig)
