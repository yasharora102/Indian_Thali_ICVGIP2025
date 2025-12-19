
<div align="center">

# Indian Thali: A Dataset and Benchmark for Food Segmentation and Weight Estimation

[![Paper](https://img.shields.io/badge/Paper-ICVGIP_2025-green)](https://cvit.iiit.ac.in/research/projects/cvit-projects/indian_thali)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-ITD_%26_WED-yellow)](https://cvit.iiit.ac.in/research/projects/cvit-projects/indian_thali)

**Yash Arora***, [Author 2], [Author 3], [Author 4]

**Abstract**
*We present the **Indian Thali Dataset (ITD)** and **Weight Estimation Dataset (WED)**, targeting the complex challenge of segmenting and estimating the weight of Indian food items. Unlike western meals, Indian thalis allow items to touch and overlap, presenting unique segmentation challenges. We also introduce a novel multi-modal weight estimation network that fuses RGB and depth features to achieve state-of-the-art results.*

[**Webpage**](https://cvit.iiit.ac.in/research/projects/cvit-projects/indian_thali) | [**Paper**](#) | [**Video**](#)

</div>

---

## 📢 News
- **[2025-12-19]** Code and datasets are released!
- **[2025-12-01]** Paper accepted at ICVGIP 2025.

---

## 📂 Repository Structure

This repository is organized into three main components:

| Component | Description |
| :--- | :--- |
| **`segmentation/`** | Training and evaluation scripts for food segmentation (Mask2Former, SegFormer, etc.) on ITD. |
| **`weight_estimation/`** | Our proposed weight estimation pipeline fusing RGB and Depth cues. |
| **`food_scanner/`** | A real-time prototype application (FastAPI) demonstrating the full pipeline. |

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the dependencies. We recommend using a Conda environment.

```bash
git clone https://github.com/yasharora102/Indian_Thali.git
cd Indian_Thali

# Create a generic env (optional but recommended)
conda create -n thali_env python=3.10 -y
conda activate thali_env

# Install core requirements
pip install -r requirements.txt
```

**Note:** The `segmentation` module relies on **MMDetection** and **MMSegmentation**. Please install them using MIM for best compatibility:
```bash
pip install -U openmim
mim install mmcv-full
mim install mmdet
mim install mmsegmentation
```

### 2. Data Preparation

Please download the datasets from our [project webpage](https://cvit.iiit.ac.in/research/projects/cvit-projects/indian_thali).

- **Indian Thali Dataset (ITD)**: Contains 7,900 annotated images.
- **Weight Estimation Dataset (WED)**: Contains ~1,400 RGB-D image pairs with ground truth weights.

Structure your data directory as follows:
```text
data/
├── ITD/
│   ├── images/
│   └── annotations/
└── WED/
    ├── rgb/
    ├── depth/
    └── weights.json
```

---

## 🛠️ Usage

### Food Segmentation
To reproduce our segmentation benchmarks (e.g., Mask2Former):

```bash
cd segmentation
# Edit paths in the script if necessary
bash examples/benchmark_mask2former.sh
```

### Weight Estimation
To train and evaluate the weight estimation network:

```bash
cd weight_estimation

# Training
bash train.sh

# Evaluation
bash evaluate.sh
```

### Food Scanner Demo
To run the interactive web demo:

```bash
cd food_scanner
# Ensure 'config.yaml' points to your models and data
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

---

## 📊 Model Zoo

We provide pre-trained checkpoints for our best performing models.

| Model | Task | Backbone | metrics | Link |
| :--- | :--- | :--- | :--- | :--- |
| **Mask2Former** | Segmentation | Swin-L | 58.2 mIoU | [Download](#) |
| **WeightNet** | Weight Est. | ResNet-50 | N/A | [Download](#) |

---

## 🖊️ Citation

If you find this code or dataset useful, please cite our work:

```bibtex
@inproceedings{icvgip2025_indian_thali,
  title={Indian Thali: A Dataset and Benchmark for Food Segmentation and Weight Estimation},
  author={Arora, Yash and ...},
  booktitle={Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP)},
  year={2025}
}
```

## 📄 License
This project is released under the [MIT License](LICENSE).
