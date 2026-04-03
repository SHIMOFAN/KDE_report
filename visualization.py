import cv2
import matplotlib.pyplot as plt


def save_mask_as_image(mask, save_path):
    """
    将掩码保存为图像文件。
    """
    cv2.imwrite(save_path, mask)


def save_mask_as_svg(mask, save_path):
    """
    将掩码保存为 SVG 格式。
    """
    plt.figure(figsize=(mask.shape[1] / 100, mask.shape[0] / 100), dpi=300)
    plt.imshow(mask, cmap='gray')
    plt.savefig(save_path)
    plt.close()


def save_image_as_svg(image, save_path):
    """
    将彩色图像保存为 SVG 格式（自动转换 BGR->RGB）。
    """
    # 如果输入是BGR格式，转为RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = image
    plt.figure(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=300)
    plt.imshow(rgb)
    plt.savefig(save_path)
    plt.close()
