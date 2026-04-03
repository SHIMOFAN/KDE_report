import numpy as np


def epanechnikov_kernel(u):
    """
    Epanechnikov 核函数。
    K(u) = 0.75 * (1 - u^2)  for |u| <= 1, else 0
    :param u: 标准化距离数组或标量
    :return: 核函数值（同形状）
    """
    kernel = 0.75 * (1 - u ** 2)
    kernel[u > 1] = 0
    return kernel


def gaussian_kernel(u, sigma=1.0):
    """
    高斯核函数（可选）。
    K(u) = (1 / (√(2π)σ)) * exp(-u²/(2σ²))
    :param u: 标准化距离
    :param sigma: 标准差（默认为1）
    :return: 核函数值
    """
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (u / sigma) ** 2)


def get_kernel(kernel_name='epanechnikov'):
    """
    根据名称返回对应的核函数。
    :param kernel_name: 'epanechnikov' 或 'gaussian'
    :return: 可调用函数
    """
    if kernel_name == 'epanechnikov':
        return epanechnikov_kernel
    elif kernel_name == 'gaussian':
        return gaussian_kernel
    else:
        raise ValueError(f"Unsupported kernel: {kernel_name}")
