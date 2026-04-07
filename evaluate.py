"""
评估前景检测结果，支持 CDnet 2014 数据集的 ROI 和 temporalROI。
计算 Precision, Recall, F1, FPR, FNR, Accuracy 等指标。
"""

import os

import cv2
import numpy as np
from tqdm import tqdm


def load_temporal_roi(temporal_roi_path):
    """
    读取 temporalROI.txt，返回起始帧和结束帧（包含）。

    Args:
        temporal_roi_path (str): temporalROI.txt 文件路径。

    Returns:
        tuple: (start_frame, end_frame) 或 (None, None) 若文件不存在。
    """
    if temporal_roi_path is None or not os.path.exists(temporal_roi_path):
        return None, None
    with open(temporal_roi_path, 'r') as f:
        line = f.readline().strip()
        start, end = map(int, line.split())
        return start, end


def load_roi_mask(roi_path, target_shape):
    """
    读取 ROI.bmp，返回对应掩码（有效区域为 True）。

    Args:
        roi_path (str): ROI.bmp 文件路径。
        target_shape (tuple): 目标图像尺寸 (h, w)。

    Returns:
        numpy.ndarray: bool 型掩码，True 表示 ROI 内像素。
    """
    if roi_path is None or not os.path.exists(roi_path):
        print("ROI文件不存在或未提供，将使用全图评估。")
        return np.ones(target_shape[:2], dtype=bool)
    roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
    if roi is None:
        print(f"无法读取 ROI 文件: {roi_path}，使用全图评估。")
        return np.ones(target_shape[:2], dtype=bool)
    return roi > 0


def evaluate_video(gt_dir, pred_dir, roi_path=None, temporal_roi_path=None):
    """
    评估视频预测结果。

    Args:
        gt_dir (str): groundtruth 目录。
        pred_dir (str): 预测结果目录。
        roi_path (str, optional): ROI.bmp 文件路径。
        temporal_roi_path (str, optional): temporalROI.txt 文件路径。

    Returns:
        dict: 包含各项指标的字典。
    """
    # 获取所有 groundtruth 文件
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith(('.bmp', '.png'))]
    gt_files.sort()

    if len(gt_files) == 0:
        raise ValueError(f"Groundtruth 目录下没有图像文件: {gt_dir}")

    # 应用 temporal ROI 过滤
    start_frame, end_frame = load_temporal_roi(temporal_roi_path)
    if start_frame is not None and end_frame is not None:
        import re
        def frame_number(filename):
            nums = re.findall(r'\d+', filename)
            return int(nums[0]) if nums else -1

        gt_files = [f for f in gt_files if start_frame <= frame_number(f) <= end_frame]

    print(f"有效评估帧数: {len(gt_files)}")

    # 读取第一帧获取图像尺寸
    first_gt = cv2.imread(os.path.join(gt_dir, gt_files[0]), cv2.IMREAD_GRAYSCALE)
    if first_gt is None:
        raise ValueError("无法读取第一帧 groundtruth")
    h, w = first_gt.shape

    # 加载 ROI 掩码
    roi_mask = load_roi_mask(roi_path, (h, w))
    n_valid_pixels = np.sum(roi_mask)
    print(f"ROI 有效像素数: {n_valid_pixels}")

    total_tp = total_fp = total_tn = total_fn = 0

    for gt_file in tqdm(gt_files, desc="Evaluating"):
        # 读取 groundtruth
        gt_path = os.path.join(gt_dir, gt_file)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        gt_binary = (gt == 255).astype(np.uint8)

        # 尝试多种可能的预测文件名（支持 _mask.png 等）
        base_name = os.path.splitext(gt_file)[0]
        candidates = [
            gt_file,
            base_name + ".png",
            base_name + "_mask.png",
            base_name + "_mask.bmp",
            base_name + "_pred.png",
            base_name + "_pred.bmp",
        ]
        pred_path = None
        for cand in candidates:
            cand_path = os.path.join(pred_dir, cand)
            if os.path.exists(cand_path):
                pred_path = cand_path
                break
        if pred_path is None:
            print(f"警告：未找到预测文件，尝试过的名称: {candidates}，跳过 {gt_file}")
            continue

        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pred is None:
            print(f"警告：无法读取预测文件 {pred_path}，跳过")
            continue
        pred_binary = (pred > 127).astype(np.uint8)

        # 只考虑 ROI 内的像素
        gt_roi = gt_binary[roi_mask]
        pred_roi = pred_binary[roi_mask]

        tp = np.sum((pred_roi == 1) & (gt_roi == 1))
        fp = np.sum((pred_roi == 1) & (gt_roi == 0))
        tn = np.sum((pred_roi == 0) & (gt_roi == 0))
        fn = np.sum((pred_roi == 0) & (gt_roi == 1))

        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

    # 计算评估指标
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    fpr = total_fp / (total_fp + total_tn + 1e-8)
    fnr = total_fn / (total_fn + total_tp + 1e-8)
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn + 1e-8)

    metrics = {
        "Precision": precision,
        "Recall": recall,
        "F1-measure": f1,
        "FPR": fpr,
        "FNR": fnr,
        "Accuracy": accuracy,
        "TP": total_tp,
        "FP": total_fp,
        "TN": total_tn,
        "FN": total_fn,
    }
    return metrics


def main():
    """命令行入口：解析参数并运行评估。"""
    import argparse
    parser = argparse.ArgumentParser(description="评估前景检测结果（支持 CDnet2014 格式）")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Groundtruth 文件夹路径（包含 frameXXXX.bmp 等）")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="预测结果文件夹路径（文件名需与 groundtruth 匹配）")
    parser.add_argument("--roi", type=str, default=None,
                        help="ROI.bmp 文件路径（若未指定则尝试在 gt_dir 中查找）")
    parser.add_argument("--temporal_roi", type=str, default=None,
                        help="temporalROI.txt 文件路径（若未指定则尝试在 gt_dir 中查找）")
    args = parser.parse_args()

    # 自动查找 ROI 和 temporalROI（如果未指定）
    roi_path = args.roi
    if roi_path is None:
        possible_roi = os.path.join(args.gt_dir, "ROI.bmp")
        if os.path.exists(possible_roi):
            roi_path = possible_roi

    temporal_path = args.temporal_roi
    if temporal_path is None:
        possible_temporal = os.path.join(args.gt_dir, "temporalROI.txt")
        if os.path.exists(possible_temporal):
            temporal_path = possible_temporal

    print("===== 开始评估 =====")
    print(f"Groundtruth 目录: {args.gt_dir}")
    print(f"Prediction 目录: {args.pred_dir}")
    print(f"ROI 文件: {roi_path if roi_path else '未使用'}")
    print(f"temporalROI 文件: {temporal_path if temporal_path else '未使用'}")

    metrics = evaluate_video(args.gt_dir, args.pred_dir, roi_path, temporal_path)

    print("\n===== 评估结果 =====")
    for key, value in metrics.items():
        if key in ["TP", "FP", "TN", "FN"]:
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
