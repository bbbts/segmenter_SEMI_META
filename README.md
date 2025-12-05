# Segmenter: Transformer for Semantic Segmentation

[Segmenter: Transformer for Semantic Segmentation](https://arxiv.org/abs/2105.05633)  
by Robin Strudel*, Ricardo Garcia*, Ivan Laptev and Cordelia Schmid, ICCV 2021.  

*Equal Contribution  

🔥 **Segmenter is now available on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/segmenter).**  

# Segmenter: Semi-Supervised Semantic Segmentation on Meta Dataset

![Segmenter Overview](./overview.png)

This repository implements **semantic segmentation using Vision Transformers (ViT)**, based on the **Segmenter architecture** ([Strudel et al., 2021](https://arxiv.org/abs/2105.05633v3)).  
It includes a **semi-supervised** setup, supporting **Meta Dataset**.

---

# 🌍 Vision Transformer (ViT) for Semantic Segmentation

**Author:** Bijoya Bhattacharjee  
**Affiliation:** Ph.D. Student, Department of Electrical and Computer Engineering, University of Nevada, Las Vegas (UNLV)  
**Research Focus:** Computer Vision & Machine Learning — Wildfire Detection & Semantic Segmentation  

---

## 📘 Table of Contents

1. [Overview](#overview)  
2. [Background & Related Works](#background--related-works)  
   - [Transformers in Vision](#transformers-in-vision)  
   - [Vision Transformer (ViT)](#vision-transformer-vit)  
   - [Segmenter: Supervised & Semi-Supervised](#segmenter-supervised--semi-supervised)  
3. [Dataset Structure](#dataset-structure)  
   - [Meta Dataset](#meta-dataset) 
4. [Installation](#installation)  
5. [Training Procedure](#training-procedure)  
6. [Evaluation, Training Logs & Plots](#evaluation-training-logs--plots)  
7. [Inference & Metrics Logging](#inference--metrics-logging)  
8. [IoU vs Labeled Dataset Script](#iou-vs-labeled-dataset-script)  
9. [Original Repo Commands](#original-repo-commands)  
10. [Repository Structure](#repository-structure)  
11. [References](#references)  
12. [Author & Acknowledgments](#author--acknowledgments)  

---

## 1️⃣ Overview

- Implements **supervised & semi-supervised semantic segmentation** using ViT backbones with Mask Transformer decoder  
- Supervised: uses **fully labeled Meta Dataset**  
- Semi-supervised: leverages **labeled + unlabeled images** with pseudo-labeling  

**Semi-supervised teacher-student mechanism:**  
- Teacher model is trained from scratch on labeled data (Meta Dataset)  
- For unlabeled images, the **teacher generates pseudo masks**, and the **student predicts masks**  
- **Student weights are updated** using the error between its prediction and the pseudo mask  
- **Teacher weights remain fixed**  

**Goal:** Dense, pixel-level segmentation for wildfire detection and general scene parsing.

---

## 2️⃣ Background & Related Works

### 🧠 Transformers in Vision
- Self-attention mechanism for sequence modeling  
- Extended to vision by splitting images into patches  

**Paper:** [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)

### 🧩 Vision Transformer (ViT)
- Split images into patches → embed as tokens  
- CLS token aggregates global info  

**Paper:** [ViT (2020)](https://arxiv.org/abs/2010.11929)  
**Code:** [Google Research ViT](https://github.com/google-research/vision_transformer)

### 🎨 Segmenter: Supervised & Semi-Supervised
- Mask Transformer decoder predicts dense masks  
- Semi-supervised setup uses pseudo-labeling for unlabeled images  
- Supports ViT Tiny, Small, Base backbones  

**Paper:** [Segmenter (2021)](https://arxiv.org/abs/2105.05633v3)  
**Code:** [https://github.com/rstrudel/segmenter](https://github.com/rstrudel/segmenter)

---

## 3️⃣ Dataset Structure

### 🔥 Meta Dataset
Meta Dataset is a **custom dataset** created to train the fully supervised teacher model. It unifies **4 different datasets**: Flame, Corsican, AWIR, and BurnedAreaUAV, and has **4 classes**: background, fire, burned area, and water.  

#### Structure
    Datasets/Meta/
    ├── images/
    │   ├── train/ (.jpg)
    │   ├── test/ (.jpg)
    │   └── validation/ (.jpg)
    └── masks/
        ├── train/ (.png)
        ├── test/ (.png)
        └── validation/ (.png)

#### Links to Original Datasets
[6] D. Sharma et al., “FLAME Dataset: Aerial Imagery for Pile-Burn Detection Using Drones (UAVs),” https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs, 2022, accessed: 2025-01-15.  
[7] University of Corsica / SPE Laboratory, “Corsican fire database (cfd),” https://cfdb.univ-corse.fr/index.php?newlang=english&menu=1, 2020, accessed: 2025-01-15.  
[8] E. Menezes et al., “AIWR: Aerial Image Water Resources Dataset,” https://data.mendeley.com/datasets/8dxv4yvvjk/1, 2021, mendeley Data, Version 1.  
[9] C. Pinto et al., “BurnedAreaUAV v1.1: UAV-Based Burned Area Segmentation Dataset,” https://zenodo.org/records/7866087, 2023, zenodo, Version 1.1.  

- Semi-supervised: additional unlabeled images can be placed in `train_unlabeled/`

---

## 4️⃣ Installation

### Clone
1. git clone [This Repo](https://github.com/bbbts/ViT_Semi_FrozenTeacher)  
2. cd ViT_Semi_FrozenTeacher

### Option 1: Conda Environment
conda create -n segmenter_env python=3.8 -y  
conda activate segmenter_env  
pip install -r requirements.txt  

### Option 2: PyTorch + pip install
1. Install [PyTorch 1.9](https://pytorch.org/)  
2. Run at repo root: pip install .

### Dataset Path
export DATASET=/path/to/Datasets/Meta

---

## 5️⃣ Training Procedure

### Semi-Supervised Training
python3 train.py \
  --dataset meta \
  --backbone vit_tiny_patch16_384 \
  --decoder mask_transformer \
  --batch-size 8 \
  --epochs 50 \
  --learning-rate 0.0001 \
  --labeled_ratio 0.5 \
  --log-dir ./logs/Meta_Semi_ViT_Tiny/

- Pseudo-labels are used for unlabeled images  
- `--labeled_ratio` controls fraction of labeled images  

---

## 6️⃣ Evaluation, Training Logs & Plots

### Training Logging (summary)
- **Loss Plot (`training_losses.png`)** shows:  
  1. Train Cross-Entropy Loss (CE)  
  2. Train Dice Loss  
  3. Supervised Loss  
  4. Unsupervised Loss  
  5. Total Loss  
  6. Validation Loss  

- **CSV Logging** contains per-epoch metrics: Pixel Accuracy, Mean Pixel Accuracy, Mean IoU, FWIoU, Per-Class F1 (Dice), Precision, Recall, Per-Class IoU.  
- All logs, PNGs and CSVs are saved to `--log-dir`.

> **Note:** The training CSV reports epoch index starting at `0`. So `epoch = 49` means the 50th epoch (0...49).

---

### Final per-epoch evaluation (supervised / semi-supervised run with `--labeled-ratio 0.5`, 50 epochs)
**Table 1 — Final evaluation metrics (epoch 49 = 50th epoch)**

| epoch | PixelAcc   | MeanAcc     | IoU         | MeanIoU     | FWIoU      | PerClassDice                                  | Precision                                      | Recall                                         | F1                                              | PerClassIoU                                  |
|-------:|-----------:|------------:|------------:|------------:|-----------:|-----------------------------------------------:|-----------------------------------------------:|-----------------------------------------------:|------------------------------------------------:|----------------------------------------------:|
| 49    | 0.99745606 | 0.90250299  | 0.82516710  | 0.82516710  | 0.99745606 | [0.99872029, 0.78999841]                       | [0.99884415, 0.77424186]                      | [0.99859643, 0.80640954]                      | [0.99871975, 0.78999788]                       | [0.99744385, 0.65289038]                     |

*(PerClassDice, Precision, Recall, F1, PerClassIoU are arrays for classes: [background, fire].)*

---

### Final loss breakdown & labeled/all splits (same run)
**Table 2 — Loss components and labeled/all performance splits**

| CE         | Weighted_CE | Dice        | Sup        | Unsup      | Total       | Validation  | PixelAcc_labeled | PixelAcc_all | IoU_labeled  | IoU_all     | Dice_labeled | Dice_all   |
|-----------:|------------:|------------:|-----------:|-----------:|------------:|------------:|-----------------:|------------:|-------------:|-----------:|-------------:|-----------:|
| 0.00810654 | 0.00810654  | 0.11966547  | 0.00810654 | 0.00338157 | 0.131153576 | 0.14670658  | 0.997546656      | 0.498773328 | 0.827817305  | 0.42892138 | 0.896282912  | 0.59696009 |

---

- **CE / Weighted_CE / Dice / Sup / Unsup / Total / Validation** — loss components logged per epoch.  
- **PixelAcc / MeanAcc / IoU / MeanIoU / FWIoU** — standard segmentation evaluation metrics.  
- **PerClassDice / PerClassIoU / Precision / Recall / F1** — per-class metrics reported as arrays in [background, fire] order.  
- **PixelAcc_labeled / PixelAcc_all, IoU_labeled / IoU_all, Dice_labeled / Dice_all** — metrics separately for labeled vs. all images.

---

## 7️⃣ Inference & Metrics Logging

### Semi-Supervised Inference
python3 inference.py \
  --image /path/to/custom_image.jpg \
  --checkpoint ./logs/Meta_Semi_ViT_Tiny/checkpoint.pth \
  --backbone vit_tiny_patch16_384 \
  --decoder mask_transformer \
  --output ./inference_results/ \
  --overlay

- Generates segmentation masks  
- `--overlay` option shows predicted mask over original image  
- **CSV metrics** include: Pixel_Acc, Mean_Acc, Mean_IoU, FWIoU, Dice, PerClassDice, Precision, Recall, F1  

---

## 8️⃣ IoU vs Labeled Dataset Script

### 📂 Directory Structure
    Working Directory/
    ├── MODEL_FILE
    │   ├── evaluation_metrics.csv
    │   ├── losses.csv
    │   ├── checkpoint.pth
    │   └── training_losses.png
    ├── MODEL_FILE_0.4
    │   ├── evaluation_metrics.csv
    │   ├── losses.csv
    │   ├── checkpoint.pth
    │   └── training_losses.png
    ├── MODEL_FILE_0.6
    │   ├── evaluation_metrics.csv
    │   ├── losses.csv
    │   ├── checkpoint.pth
    │   └── training_losses.png
    ├── MODEL_FILE_0.7
    │   ├── evaluation_metrics.csv
    │   ├── losses.csv
    │   ├── checkpoint.pth
    │   └── training_losses.png
    ├── PREDICTION/meta
    │   ├── *.jpg predictions
    │   └── eval_metrics.csv
    ├── PREDICTION_0.4/meta
    │   ├── *.jpg predictions
    │   └── eval_metrics.csv
    ├── PREDICTION_0.6/meta
    │   ├── *.jpg predictions
    │   └── eval_metrics.csv
    └── PREDICTION_0.7/meta
        ├── *.jpg predictions
        └── eval_metrics.csv

### ⚙️ How `iou_vs_label.py` Works
- Reads `eval_metrics.csv` and `losses.csv` from multiple trained models (with different labeled ratios)  
- Generates **two plots**:
  1. **Mean IoU vs Labeled Dataset Ratio** – shows segmentation performance vs fraction of labeled data  
  2. **Training Loss vs Labeled Dataset Ratio** – shows final training loss vs fraction of labeled data  
- Automatically matches `PREDICTION*` folders to `MODEL_FILE*` folders, extracts metrics, and saves plots to the output directory


✅ **Usage Example**

    python3 iou_vs_label.py \
      --predictions-root /home/AD.UNLV.EDU/bhattb3/segmenter_SEMI1/segm/ \
      --models-root /home/AD.UNLV.EDU/bhattb3/segmenter_SEMI1/segm/ \
      --output-dir /home/AD.UNLV.EDU/bhattb3/segmenter_SEMI1/segm/plots/

This will generate the following plots in your specified output directory:
- mIoU_vs_labeled_ratio.png
- loss_vs_labeled_ratio.png
We also generate a Feature similarty (internal transformer embeddings) vs labeled dataset ratio plot to quantify how the model's inernal representation changes with increase in labeled training data. 

**Example Plots:**

<p align="center">
  <img src="./loss_vs_labeled_ratio.png" alt="Loss vs Labeled Dataset Ratio" width="450">
</p>

<p align="center">
  <img src="./feature_similarity_trend.png" alt="Feature similarity vs Labeled Dataset Ratio" width="450">
</p>

---

## 9️⃣ Original Repo Commands

### Inference
    python -m segm.inference --model-path seg_tiny_mask/checkpoint.pth -i images/ -o segmaps/

### ADE20K Evaluation
Single-scale evaluation:
    python -m segm.eval.miou seg_tiny_mask/checkpoint.pth ade20k --singlescale

Multi-scale evaluation:
    python -m segm.eval.miou seg_tiny_mask/checkpoint.pth ade20k --multiscale

### Training (ADE20K)
    python -m segm.train --log-dir seg_tiny_mask --dataset ade20k \
      --backbone vit_tiny_patch16_384 --decoder mask_transformer

Note: For `Seg-B-Mask/16` use `vit_base_patch16_384` and ≥4 V100 GPUs.

### Logs
    python -m segm.utils.logs logs.yml

Example `logs.yml`:
    root: /path/to/checkpoints/
    logs:
      seg-t: seg_tiny_mask/log.txt
      seg-b: seg_base_mask/log.txt

---

## 🔟 Repository Structure

segmenter-meta/
├── segm/                    # Core Segmenter code  
├── train.py                 # Supervised training  
├── train_semi.py            # Semi-supervised training  
├── eval.py                  # Evaluation script  
├── inference.py             # Supervised inference  
├── inference_semi.py        # Semi-supervised inference  
├── iou_vs_label.py          # mIoU vs Labeled Dataset script  
├── requirements.txt         # Dependencies  
├── datasets/                # Dataset loaders  
├── logs/                    # Checkpoints, plots, CSV logs  
├── README.md                # Project documentation  
└── utils/                   # Helper scripts  

---

## 🔟 References

| Year | Paper | Authors | Link |
|------|-------|---------|------|
| 2017 | *Attention Is All You Need* | Vaswani et al. | https://arxiv.org/abs/1706.03762 |
| 2020 | *An Image is Worth 16x16 Words* | Dosovitskiy et al. | https://arxiv.org/abs/2010.11929 |
| 2021 | *Segmenter: Transformer for Semantic Segmentation* | Strudel et al. | https://arxiv.org/abs/2105.05633v3 |
| 2021 | *Segmenter GitHub* | Strudel et al. | https://github.com/rstrudel/segmenter |
| 2022 | *FLAME: Fire Segmentation Dataset* | IEEE Dataport | https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs |
| 2020 | *Corsican fire database (cfd)* | University of Corsica | https://cfdb.univ-corse.fr/index.php?newlang=english&menu=1 |
| 2021 | *AIWR: Aerial Image Water Resources Dataset* | Menezes et al. | https://data.mendeley.com/datasets/8dxv4yvvjk/1 |
| 2023 | *BurnedAreaUAV v1.1* | Pinto et al. | https://zenodo.org/records/7866087 |
| 2017 | *ADE20K Benchmark* | Zhou et al. | https://groups.csail.mit.edu/vision/datasets/ADE20K/ |

---

## 🔟 Author & Acknowledgments

**Author:**  
**Bijoya Bhattacharjee**  
Ph.D. Candidate, Department of Electrical and Computer Engineering, University of Nevada, Las Vegas (UNLV)

**Research Interests:**  
- Advanced Computer Vision techniques for wildfire detection and monitoring  
- Semantic segmentation using Vision Transformers (ViT)  
- Semi-supervised and self-supervised learning for remote sensing applications  
- Multimodal and multispectral data integration for environmental monitoring  

**Acknowledgments:**  
This work builds upon the **Segmenter framework** (Strudel et al., 2021) and leverages open-source tools including **timm** and **MMSegmentation**. The semi-supervised teacher-student setup enables the model to learn from unlabeled UAV imagery, reducing the need for extensive manual annotations while maintaining high segmentation accuracy.  

> “By harnessing both labeled and unlabeled data, Vision Transformers can capture richer spatial features for accurate wildfire segmentation, pushing the boundaries of automated environmental monitoring.”

