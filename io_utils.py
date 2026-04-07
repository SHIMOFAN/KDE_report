"""
图像文件 I/O 工具函数。
提供获取文件夹内图像列表、读取灰度/彩色图像的功能。
"""

import os
from glob import glob

import cv2


def get_image_files(folder, extensions=None):
    """
    获取指定文件夹中所有图像文件的路径列表。

    Args:
        folder (str): 文件夹路径。
        extensions (list, optional): 支持的扩展名列表，默认为 ['.jpg']。

    Returns:
        list: 排序后的文件路径列表。
    """
    if extensions is None:
        extensions = ['.jpg']
    if isinstance(extensions, str):
        extensions = [extensions]

    images = []
    for ext in extensions:
        # 同时匹配小写和大写扩展名
        images.extend(glob(os.path.join(folder, f'*{ext}')))
        images.extend(glob(os.path.join(folder, f'*{ext.upper()}')))
    images.sort()
    return images


def read_image_grayscale(image_path):
    """
    以灰度模式读取图像。

    Args:
        image_path (str): 图像文件路径。

    Returns:
        numpy.ndarray or None: 灰度图像数组，读取失败返回 None。
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img


def read_image_color(image_path):
    """
    以彩色模式读取图像（BGR 格式）。

    Args:
        image_path (str): 图像文件路径。

    Returns:
        numpy.ndarray or None: BGR 图像数组，读取失败返回 None。
    """
    img = cv2.imread(image_path)
    return img
