import os

from core import KDEBackgroundDetector
from io_utils import get_image_files, read_image_color
from visualization import (
    save_mask_as_image,
    save_mask_as_svg,
    save_image_as_svg
)


def main():
    train_path = r"./train"
    test_path = r"./test"
    output_path = "gaussian"
    bandwidth = 30
    threshold = 1e-6
    kernel = 'gaussian'

    detector = KDEBackgroundDetector(bandwidth, threshold, kernel)

    if not detector.train(train_path):
        print("训练失败！")
        return

    test_images = get_image_files(test_path)
    if not test_images:
        print(f"测试文件夹 {test_path} 中没有图像")
        return

    os.makedirs(output_path, exist_ok=True)

    print("正在进行前景检测...")
    for img_path in test_images:
        img = read_image_color(img_path)
        if img is None:
            continue

        mask = detector.detect_single_frame_vectorized(img)

        basename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(basename)[0]

        # 保存掩码图像
        mask_img_path = os.path.join(output_path, f"mask_{basename}")
        save_mask_as_image(mask, mask_img_path)

        # 保存掩码SVG
        mask_svg_path = os.path.join(output_path, f"mask_{name_no_ext}.svg")
        save_mask_as_svg(mask, mask_svg_path)

        # 保存原图SVG
        orig_svg_path = os.path.join(output_path, f"{name_no_ext}.svg")
        save_image_as_svg(img, orig_svg_path)

        # ========== 为当前图像绘制三维KDE图 ==========
        kde_svg_path = os.path.join(output_path, f"kde_{name_no_ext}.svg")
        detector.plot_3d_kde(
            frame=img,
            step=3,
            elev=60,
            azim=30,
            cmap='viridis',
            use_log_scale=True,
            plot_type='surface',
            z_compression=0.15,
            save_path=kde_svg_path
        )
        # ==========================================

    print("检测完成！")


if __name__ == "__main__":
    main()
