
<div align="center">

# What is there in an Indian Thali?

[![Paper](https://img.shields.io/badge/Paper-ICVGIP_2025-green)](https://drive.google.com/file/d/1UpYiVOng2okzTeanZ2ds1G7EpEnONfb_/view?usp=sharing)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-ITD_%26_WED-yellow)](https://cvit.iiit.ac.in/research/projects/cvit-projects/indian_thali)

**Yash Arora**, **Aditya Arun**, **C. V. Jawahar**

**International Institute of Information Technology, Hyderabad (IIIT-H)**

**Abstract**
*Automated dietary monitoring solutions face significant challenges when dealing with culturally diverse, multi-dish meals, where traditional single-item recognition approaches fail to capture the complexity of real-world eating patterns. Most existing computer vision systems are tailored to western foods and struggle with the overlapping textures, varied presentations, and cultural specificity of dishes like Indian Thalis, which contain 5–10 distinct food items per plate. We present Food Scanner, a novel, end-to-end pipeline with retraining-free segmentation & prototype-based classification, plus a lightweight trainable weight-regression head for automated nutrition estimation of multi-dish meals from a single image. Our approach requires no class-specific segmentation or classification retraining, enabling rapid adaptation to new dishes and cuisines. The pipeline integrates zero-shot segmentation, embedding-based prototype classification, a lightweight weight regression head, and nutrition computation to transform an Indian thali into per-dish calorie and macronutrient breakdowns. To enable this study, we contribute two datasets: a multi-view Indian Thali dataset of 796 plates (7,900 images) covering 50 dishes (with dense plate-level masks), and a weight estimation dataset of 267 plates (1,394 images) covering 41 dishes (with gram-level weight annotations). Systematic ablation studies show that our method achieves high accuracy while maintaining real-time performance. By combining zero-shot capabilities with a modular design, Food Scanner offers a scalable, culturally adaptable solution that can be deployed across diverse food environments without any additional training. The code will be available here.*

[**Webpage**](https://cvit.iiit.ac.in/research/projects/cvit-projects/indian_thali) | [**Paper**](https://drive.google.com/file/d/1UpYiVOng2okzTeanZ2ds1G7EpEnONfb_/view?usp=sharing)

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
git clone https://github.com/yasharora102/Indian_Thali_ICVGIP2025.git
cd Indian_Thali_ICVGIP2025

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
| **Mask2Former** | Segmentation | Swin-L | 58.2 mIoU | Coming Soon |
| **WeightNet** | Weight Est. | ResNet-50 | N/A | Coming Soon |

---

## 🖊️ Citation

If you find this code or dataset useful, please cite our paper:

```bibtex
@inproceedings{arora2025indian,
  title={What is there in an Indian Thali?},
  author={Arora, Yash and Arun, Aditya and Jawahar, C V},
  booktitle={Proceedings of the Indian Conference on Computer Vision, Graphics, and Image Processing (ICVGIP)},
  year={2025}
}
```

## 📄 License
This project is released under the [MIT License](LICENSE).
