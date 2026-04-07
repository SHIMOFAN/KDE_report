"""
可视化与图像保存工具。
提供掩码保存、SVG 格式输出等功能。
"""

import cv2
import matplotlib.pyplot as plt


def save_mask_as_image(mask, save_path):
    """
    将掩码保存为图像文件（PNG/JPG 等）。

    Args:
        mask (numpy.ndarray): 二值掩码（0 或 255）。
        save_path (str): 保存路径。
    """
    cv2.imwrite(save_path, mask)


def save_mask_as_svg(mask, save_path):
    """
    将掩码保存为 SVG 矢量格式。

    Args:
        mask (numpy.ndarray): 二值掩码。
        save_path (str): 保存路径（建议以 .svg 结尾）。
    """
    plt.figure(figsize=(mask.shape[1] / 100, mask.shape[0] / 100), dpi=300)
    plt.imshow(mask, cmap='gray')
    plt.savefig(save_path)
    plt.close()


def save_image_as_svg(image, save_path):
    """
    将彩色图像保存为 SVG 格式（自动转换 BGR->RGB）。

    Args:
        image (numpy.ndarray): BGR 格式彩色图像。
        save_path (str): 保存路径（建议以 .svg 结尾）。
    """
    # 转换 BGR 到 RGB（matplotlib 期望 RGB）
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = image
    plt.figure(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=300)
    plt.imshow(rgb)
    plt.savefig(save_path)
    plt.close()
