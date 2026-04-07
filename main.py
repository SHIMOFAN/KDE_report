"""
KDE 背景建模与前景检测主程序。
支持训练背景模型、测试视频序列、后处理（形态学、面积滤波、形状筛选）。
"""

import argparse
import os
import shutil
import tempfile

import cv2
from tqdm import tqdm

from core import KDEBackgroundDetector
from io_utils import get_image_files
from visualization import save_mask_as_image


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="KDE 背景建模与前景检测")
    parser.add_argument("--train_dir", type=str, required=True,
                        help="训练图像文件夹路径（用于背景建模）")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="测试图像文件夹路径（可选，若不提供则仅训练）")
    parser.add_argument("--out_dir", type=str, default="./output",
                        help="保存检测结果的文件夹")
    parser.add_argument("--bandwidth", type=float, default=30,
                        help="核带宽（像素值范围0-255）")
    parser.add_argument("--threshold", type=float, default=1e-6,
                        help="前景判定概率阈值")
    parser.add_argument("--kernel", type=str, default="epanechnikov",
                        choices=["epanechnikov", "gaussian"],
                        help="核函数类型")
    parser.add_argument("--extensions", type=str, nargs="+",
                        default=[".jpg", ".png", ".jpeg"],
                        help="图像文件扩展名列表")
    parser.add_argument("--train_frames", type=int, default=-1,
                        help="使用的训练帧数（-1表示使用全部）。建议设为50（纯背景帧数）")
    parser.add_argument("--morph_open", action="store_true",
                        help="对预测结果进行形态学开运算（去除小噪点）")
    parser.add_argument("--morph_close", action="store_true",
                        help="对预测结果进行形态学闭运算（填充小洞，连接邻近区域）")
    parser.add_argument("--min_area", type=int, default=0,
                        help="面积滤波阈值：只保留面积大于该值的连通区域（0表示不滤波）。建议30-50")
    parser.add_argument("--shape_filter", action="store_true",
                        help="启用连通域形状筛选（过滤不符合行人宽高比和实心度的区域）")
    return parser.parse_args()


def prepare_training_images(train_dir, extensions, max_frames):
    """
    准备训练图像列表。若 max_frames > 0 且图像数量超过该值，则只取前 max_frames 张，
    并复制到临时目录（避免修改原始数据）。

    Args:
        train_dir (str): 原始训练图像文件夹。
        extensions (list): 支持的扩展名。
        max_frames (int): 最大使用帧数，-1 表示全部。

    Returns:
        str: 实际使用的训练目录路径（可能是临时目录）。
    """
    all_images = get_image_files(train_dir, extensions)
    if len(all_images) == 0:
        raise ValueError(f"训练文件夹中无有效图像：{train_dir}")

    if max_frames > 0 and len(all_images) > max_frames:
        selected = all_images[:max_frames]
        temp_dir = tempfile.mkdtemp(prefix="kde_train_")
        print(f"只使用前 {max_frames} 张图像训练，临时目录：{temp_dir}")
        for src_path in selected:
            dst_path = os.path.join(temp_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
        return temp_dir
    else:
        return train_dir  # 使用原始目录


def apply_area_filter(mask, min_area):
    """
    移除面积小于 min_area 的连通区域。

    Args:
        mask (numpy.ndarray): 二值掩码。
        min_area (int): 最小面积阈值。

    Returns:
        numpy.ndarray: 过滤后的掩码。
    """
    if min_area <= 0:
        return mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    return mask


def apply_shape_filter(mask):
    """
    过滤不符合行人形状特征的连通区域。
    行人通常具有合理的宽高比（0.3~1.5）和实心度（面积/外接矩形面积 > 0.4）。

    Args:
        mask (numpy.ndarray): 二值掩码。

    Returns:
        numpy.ndarray: 过滤后的掩码。
    """
    min_aspect_ratio = 0.3
    max_aspect_ratio = 1.5
    min_solidity = 0.4

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect_ratio = w / h
        area = cv2.contourArea(cnt)
        rect_area = w * h
        solidity = area / rect_area if rect_area > 0 else 0
        if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and solidity >= min_solidity):
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    return mask


def main():
    """主函数：训练检测器并（可选）对测试集进行检测。"""
    args = parse_args()

    # 1. 初始化检测器
    detector = KDEBackgroundDetector(
        bandwidth=args.bandwidth,
        threshold=args.threshold,
        kernel=args.kernel
    )

    # 2. 准备训练图像（必要时使用临时目录）
    print("===== 开始训练 =====")
    effective_train_dir = prepare_training_images(
        args.train_dir, args.extensions, args.train_frames
    )
    is_temp_dir = (effective_train_dir != args.train_dir)

    try:
        success = detector.train(effective_train_dir, frame_extensions=args.extensions)
        if not success:
            print("训练失败，请检查训练文件夹路径或图像格式。")
            return
    finally:
        # 如果使用了临时目录，训练完成后删除
        if is_temp_dir:
            shutil.rmtree(effective_train_dir)
            print("已删除临时训练目录")

    # 3. 如果提供了测试文件夹，进行检测
    if args.test_dir is not None:
        if not os.path.isdir(args.test_dir):
            print(f"测试文件夹不存在：{args.test_dir}")
            return

        test_images = get_image_files(args.test_dir, args.extensions)
        if len(test_images) == 0:
            print(f"测试文件夹中无有效图像：{args.test_dir}")
            return

        os.makedirs(args.out_dir, exist_ok=True)

        print(f"\n===== 开始检测 {len(test_images)} 张测试图像 =====")
        for idx, img_path in enumerate(tqdm(test_images)):
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"警告：无法读取 {img_path}，跳过")
                continue

            # 前景检测
            mask = detector.detect_single_frame_vectorized(frame)

            # 后处理1：形态学开运算（去除小噪点）
            if args.morph_open:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # 后处理2：形态学闭运算（填充小洞，连接邻近区域）
            if args.morph_close:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 后处理3：面积滤波
            if args.min_area > 0:
                mask = apply_area_filter(mask, args.min_area)

            # 后处理4：形状筛选（过滤非行人形状）
            if args.shape_filter:
                mask = apply_shape_filter(mask)

            # 生成输出文件名：将 "in" 替换为 "gt"，后缀固定为 .png
            base_name = os.path.basename(img_path)
            base_name = base_name.replace("in", "gt", 1)
            name_without_ext, _ = os.path.splitext(base_name)
            out_path = os.path.join(args.out_dir, f"{name_without_ext}.png")
            save_mask_as_image(mask, out_path)

        print(f"检测完成，结果保存在：{args.out_dir}")
    else:
        print("未提供测试文件夹，训练完成。")

    print("\n===== 任务结束 =====")


if __name__ == "__main__":
    main()
