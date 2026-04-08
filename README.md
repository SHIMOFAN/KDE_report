# KDE_report
<div align="center">

# KDE-Background-Detector

**基于核密度估计的运动目标检测 | Moving Object Detection via Kernel Density Estimation**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-CDnet%202014-orange)](http://changedetection.net/)

[English](#english) | [中文](#中文)

</div>

---

<a name="中文"></a>

## 中文文档

### 项目简介

本项目实现了一种**基于核密度估计（KDE）的视频背景建模与运动目标检测**系统，支持 Epanechnikov 核与高斯核两种核函数。通过对训练帧的逐像素灰度历史建立非参数概率密度模型，以密度阈值实现前景/背景二值分离，并提供三维概率密度可视化功能。

本项目为东华大学《机器学习》课程第一次大作业的配套代码，论文题目为：

> **基于 Epanechnikov 核函数三维核密度估计的运动目标二值检测研究**

### 核心特性

- 🔲 **双核函数支持**：Epanechnikov 核（紧支撑，计算高效）与高斯核（无限支撑，平滑性好）
- ⚡ **向量化推理**：基于 NumPy 广播机制，全帧检测无循环
- 🎛️ **丰富的后处理**：形态学开/闭运算、面积滤波、行人形状筛选
- 📊 **三维KDE可视化**：支持 Surface / Scatter 模式、对数缩放、Z轴压缩因子
- 📋 **标准评估**：兼容 CDnet 2014 数据集格式，计算 Precision / Recall / F1 / FPR / FNR / Accuracy

### 文件结构

```
KDE-Background-Detector/
├── core.py              # 核心检测器：背景建模、前景检测、3D可视化
├── kernels.py           # 核函数定义：Epanechnikov 核、高斯核
├── io_utils.py          # 图像 I/O 工具：文件列举、灰度/彩色读取
├── visualization.py     # 可视化工具：掩码/图像的 PNG/SVG 保存
├── main.py              # 主程序：命令行训练与检测入口
├── evaluate.py          # 评估脚本：基于 CDnet 格式的指标计算
└── README.md
```

### 环境依赖

```bash
pip install numpy opencv-python matplotlib tqdm
```

| 依赖包        | 推荐版本 | 用途             |
| ------------- | -------- | ---------------- |
| numpy         | ≥ 1.21   | 向量化计算       |
| opencv-python | ≥ 4.5    | 图像读取与形态学 |
| matplotlib    | ≥ 3.4    | 三维密度可视化   |
| tqdm          | ≥ 4.0    | 训练/检测进度条  |

### 快速开始

#### 1. 训练背景模型并检测前景

```bash
python main.py \
  --train_dir  /path/to/train/frames \
  --test_dir   /path/to/test/frames \
  --out_dir    ./output \
  --bandwidth  30 \
  --threshold  1e-6 \
  --kernel     epanechnikov \
  --train_frames 50
```

#### 2. 启用后处理（推荐）

```bash
python main.py \
  --train_dir  /path/to/train/frames \
  --test_dir   /path/to/test/frames \
  --out_dir    ./output \
  --bandwidth  30 \
  --threshold  1e-6 \
  --morph_open \
  --morph_close \
  --min_area   40 \
  --shape_filter
```

#### 3. 评估检测结果（CDnet 2014 格式）

```bash
python evaluate.py \
  --gt_dir       /path/to/groundtruth \
  --pred_dir     ./output \
  --roi          /path/to/ROI.bmp \
  --temporal_roi /path/to/temporalROI.txt
```

### 参数说明

#### `main.py` 参数

| 参数             | 类型  | 默认值            | 说明                                    |
| ---------------- | ----- | ----------------- | --------------------------------------- |
| `--train_dir`    | str   | 必填              | 训练帧文件夹路径                        |
| `--test_dir`     | str   | `None`            | 测试帧文件夹路径（不填则仅训练）        |
| `--out_dir`      | str   | `./output`        | 检测结果保存路径                        |
| `--bandwidth`    | float | `30`              | 核带宽（像素值范围 0–255）              |
| `--threshold`    | float | `1e-6`            | 前景判定密度阈值，越小越严格            |
| `--kernel`       | str   | `epanechnikov`    | 核函数类型：`epanechnikov` / `gaussian` |
| `--train_frames` | int   | `-1`（全部）      | 使用的训练帧数，建议取50（纯背景帧数）  |
| `--extensions`   | list  | `.jpg .png .jpeg` | 图像文件扩展名                          |
| `--morph_open`   | flag  | 关闭              | 形态学开运算（去除小噪点）              |
| `--morph_close`  | flag  | 关闭              | 形态学闭运算（填充空洞，连接近邻区域）  |
| `--min_area`     | int   | `0`（不滤波）     | 面积滤波阈值，建议 30–50                |
| `--shape_filter` | flag  | 关闭              | 行人宽高比 + 实心度形状筛选             |

#### `evaluate.py` 参数

| 参数             | 类型 | 说明                                        |
| ---------------- | ---- | ------------------------------------------- |
| `--gt_dir`       | str  | Groundtruth 目录（`.bmp` 或 `.png`）        |
| `--pred_dir`     | str  | 预测结果目录（文件名需与 GT 匹配）          |
| `--roi`          | str  | `ROI.bmp` 路径（不填则自动在 gt_dir 查找）  |
| `--temporal_roi` | str  | `temporalROI.txt` 路径（同上）              |

### 方法原理

给定训练集中 $N$ 帧的像素灰度值历史 $\{x_1, x_2, \ldots, x_N\}$，对当前帧像素 $x$ 的概率密度估计为：

$$\hat{f}(x) = \frac{1}{N \cdot h} \sum_{i=1}^{N} K\!\left(\frac{x - x_i}{h}\right)$$

其中 $h$ 为带宽，$K(\cdot)$ 为核函数。当 $\hat{f}(x) < \tau$（阈值）时，该像素判定为**前景**。

**Epanechnikov 核**（本项目推荐）：

$$K(u) = \frac{3}{4}(1 - u^2) \cdot \mathbf{1}_{|u| \leq 1}$$

具有紧支撑性，计算量显著低于高斯核。

### 数据集

本项目在以下数据集上进行了实验：

- **课程提供数据集**：包含交通场景视频序列
- **[CDnet 2014](http://changedetection.net/)**：变化检测公开基准，包含 background、dynamic background、intermittent object motion 等多个类别

数据集目录结构建议：

```
dataset/
├── input/          # 输入视频帧（frameXXXXXX.jpg）
├── groundtruth/    # 标注掩码（gtXXXXXX.png 或 .bmp）
├── ROI.bmp         # 感兴趣区域掩码
└── temporalROI.txt # 有效评估帧范围（起始帧 结束帧）
```

### 评估指标

| 指标       | 公式                                   |
| ---------- | -------------------------------------- |
| Precision  | $TP / (TP + FP)$                       |
| Recall     | $TP / (TP + FN)$                       |
| F1-measure | $2 \cdot P \cdot R / (P + R)$          |
| FPR        | $FP / (FP + TN)$                       |
| FNR        | $FN / (FN + TP)$                       |
| Accuracy   | $(TP + TN) / (TP + FP + TN + FN)$      |

### 已知局限

- 带宽 $h$ 为全局固定值，对光照剧烈变化场景适应性有限
- 背景模型为静态（不支持在线更新），对动态背景场景效果较差
- 单通道灰度建模，忽略颜色信息

### 引用

如果本项目对您的研究有所帮助，欢迎引用：

```bibtex
@misc{shi2026kde,
  author = {石墨凡},
  title  = {基于 Epanechnikov 核函数三维核密度估计的运动目标二值检测},
  year   = {2026},
  url    = {https://github.com/SHIMOFAN/KDE_report}
}
```

---

<a name="english"></a>

## English Documentation

### Overview

This project implements a **Kernel Density Estimation (KDE)-based video background modeling and moving object detection** system. It supports both the Epanechnikov kernel and the Gaussian kernel. A non-parametric per-pixel probability density model is built from the grayscale history of training frames, and foreground/background binary separation is achieved by thresholding the estimated density. A 3D probability density surface visualization is also included.

This codebase accompanies the course project for *Machine Learning* at Donghua University:

> **A Study on Binary Detection of Moving Objects Using 3D KDE Based on the Epanechnikov Kernel**

### Features

- 🔲 **Dual kernel support**: Epanechnikov (compact support, computationally efficient) and Gaussian (infinite support, smooth)
- ⚡ **Vectorized inference**: Full-frame detection with NumPy broadcasting — no per-pixel loops
- 🎛️ **Rich post-processing**: morphological open/close, area filtering, pedestrian shape filtering
- 📊 **3D KDE visualization**: Surface / Scatter modes, log-scale Z-axis, Z-compression factor
- 📋 **Standard evaluation**: CDnet 2014-compatible metrics (Precision / Recall / F1 / FPR / FNR / Accuracy)

### File Structure

```
KDE-Background-Detector/
├── core.py              # Core detector: background modeling, detection, 3D visualization
├── kernels.py           # Kernel functions: Epanechnikov and Gaussian
├── io_utils.py          # Image I/O: file listing, grayscale/color reading
├── visualization.py     # Visualization: save masks/images as PNG or SVG
├── main.py              # Entry point: CLI training and detection pipeline
├── evaluate.py          # Evaluation script: CDnet-format metric computation
└── README.md
```

### Requirements

```bash
pip install numpy opencv-python matplotlib tqdm
```

| Package        | Recommended | Purpose                     |
| -------------- | ----------- | --------------------------- |
| numpy          | ≥ 1.21      | Vectorized computation      |
| opencv-python  | ≥ 4.5       | Image I/O and morphology    |
| matplotlib     | ≥ 3.4       | 3D density surface plotting |
| tqdm           | ≥ 4.0       | Progress bars               |

### Quick Start

#### 1. Train background model and detect foreground

```bash
python main.py \
  --train_dir  /path/to/train/frames \
  --test_dir   /path/to/test/frames \
  --out_dir    ./output \
  --bandwidth  30 \
  --threshold  1e-6 \
  --kernel     epanechnikov \
  --train_frames 50
```

#### 2. With post-processing enabled (recommended)

```bash
python main.py \
  --train_dir  /path/to/train/frames \
  --test_dir   /path/to/test/frames \
  --out_dir    ./output \
  --bandwidth  30 \
  --threshold  1e-6 \
  --morph_open \
  --morph_close \
  --min_area   40 \
  --shape_filter
```

#### 3. Evaluate detection results (CDnet 2014 format)

```bash
python evaluate.py \
  --gt_dir       /path/to/groundtruth \
  --pred_dir     ./output \
  --roi          /path/to/ROI.bmp \
  --temporal_roi /path/to/temporalROI.txt
```

### Parameters

#### `main.py`

| Argument         | Type  | Default           | Description                                        |
| ---------------- | ----- | ----------------- | -------------------------------------------------- |
| `--train_dir`    | str   | required          | Path to training frames folder                     |
| `--test_dir`     | str   | `None`            | Path to test frames folder (omit to train only)    |
| `--out_dir`      | str   | `./output`        | Output directory for detection results             |
| `--bandwidth`    | float | `30`              | Kernel bandwidth (pixel value range 0–255)         |
| `--threshold`    | float | `1e-6`            | Foreground density threshold (lower = stricter)    |
| `--kernel`       | str   | `epanechnikov`    | Kernel type: `epanechnikov` or `gaussian`          |
| `--train_frames` | int   | `-1` (all)        | Number of training frames; ~50 for pure background |
| `--extensions`   | list  | `.jpg .png .jpeg` | Image file extensions                              |
| `--morph_open`   | flag  | off               | Morphological opening (removes small noise)        |
| `--morph_close`  | flag  | off               | Morphological closing (fills holes, bridges gaps)  |
| `--min_area`     | int   | `0` (off)         | Area filter threshold; recommend 30–50             |
| `--shape_filter` | flag  | off               | Pedestrian aspect ratio + solidity shape filter    |

#### `evaluate.py`

| Argument         | Type | Description                                                |
| ---------------- | ---- | ---------------------------------------------------------- |
| `--gt_dir`       | str  | Groundtruth directory (`.bmp` or `.png` frames)           |
| `--pred_dir`     | str  | Prediction directory (filenames must match GT)             |
| `--roi`          | str  | Path to `ROI.bmp` (auto-searched in `gt_dir` if omitted)  |
| `--temporal_roi` | str  | Path to `temporalROI.txt` (same auto-search behavior)     |

### Method

Given the grayscale history $\{x_1, \ldots, x_N\}$ from $N$ training frames at a pixel location, the KDE estimate for the current frame pixel $x$ is:

$$\hat{f}(x) = \frac{1}{N \cdot h} \sum_{i=1}^{N} K\!\left(\frac{x - x_i}{h}\right)$$

where $h$ is the bandwidth and $K(\cdot)$ is the kernel function. A pixel is classified as **foreground** when $\hat{f}(x) < \tau$.

**Epanechnikov kernel** (recommended in this project):

$$K(u) = \frac{3}{4}(1 - u^2) \cdot \mathbf{1}_{|u| \leq 1}$$

Its compact support makes it significantly faster than the Gaussian kernel with no meaningful loss of detection quality.

### Datasets

Experiments were conducted on:

- **Course-provided dataset**: Traffic scene video sequences
- **[CDnet 2014](http://changedetection.net/)**: A public change detection benchmark covering categories such as baseline, dynamic background, and intermittent object motion

Recommended dataset directory layout:

```
dataset/
├── input/          # Input video frames (frameXXXXXX.jpg)
├── groundtruth/    # Annotation masks (gtXXXXXX.png or .bmp)
├── ROI.bmp         # Region-of-interest mask
└── temporalROI.txt # Valid evaluation frame range (start end)
```

### Evaluation Metrics

| Metric     | Formula                               |
| ---------- | ------------------------------------- |
| Precision  | $TP / (TP + FP)$                      |
| Recall     | $TP / (TP + FN)$                      |
| F1-measure | $2 \cdot P \cdot R / (P + R)$         |
| FPR        | $FP / (FP + TN)$                      |
| FNR        | $FN / (FN + TP)$                      |
| Accuracy   | $(TP + TN) / (TP + FP + TN + FN)$     |

### Known Limitations

- The bandwidth $h$ is a **global fixed value** — limited adaptability to scenes with drastic illumination changes
- The background model is **static** (no online update), reducing robustness on dynamic background scenes
- **Single-channel grayscale** modeling ignores color information

### Citation

If you find this project useful, please consider citing:

```bibtex
@misc{shi2026kde,
  author = {Shi, Mofan},
  title  = {Binary Detection of Moving Objects Using 3D KDE Based on the Epanechnikov Kernel},
  year   = {2026},
  url    = {https://github.com/SHIMOFAN/KDE_report}
}
```

### License

This project is released under the [MIT License](LICENSE).
