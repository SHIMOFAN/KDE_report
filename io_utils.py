import os
from glob import glob

import cv2


def get_image_files(folder, extensions=None):
    """
    获取指定文件夹中所有图像文件的路径列表。
    :param folder: 文件夹路径
    :param extensions: 支持的扩展名列表，默认 .jpg
    :return: 排序后的文件路径列表
    """
    if extensions is None:
        extensions = '.jpg'
    if isinstance(extensions, str):
        extensions = [extensions]

    images = []
    for ext in extensions:
        images.extend(glob(os.path.join(folder, f'*{ext}')))
        images.extend(glob(os.path.join(folder, f'*{ext.upper()}')))
    images.sort()
    return images


def read_image_grayscale(image_path):
    """
    以灰度模式读取图像，返回numpy数组；读取失败返回None。
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img


def read_image_color(image_path):
    """
    以彩色模式读取图像，返回BGR格式numpy数组；读取失败返回None。
    """
    img = cv2.imread(image_path)
    return img
