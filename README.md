# Semi-Supervised Teacher–Student Segmentation (Meta Dataset)
This repository implements a Teacher–Student semi-supervised learning framework for semantic segmentation using the Meta Dataset, which contains four wildfire-related classes collected from multiple sources.

## Overview
This project trains:
1. Teacher Model (Fully Supervised)
   - Trained only with labeled data from the Meta Dataset.
2. Student Model (Semi-Supervised)
   - Trained using both labeled and unlabeled data.
   - Uses teacher-generated pseudo-labels and consistency regularization.

Meta Dataset = Flame + Corsican + AIWR + BurnedAreaUAV (4 classes):
- Fire
- Smoke
- Burned Area
- Background

## Directory Structure
project_root/
    configs/
        teacher_config.yaml
        student_config.yaml
    data/
        meta_dataset/
            images/
            masks/
            unlabeled/
    models/
        teacher/
        student/
    scripts/
        train_teacher.py
        train_student.py
        generate_pseudolabels.py
    utils/
        dataset_loader.py
        loss_functions.py
        helpers.py
    outputs/
        teacher_checkpoints/
        student_checkpoints/
        logs/

## Training Pipeline

### 1. Train Teacher (Fully Supervised)
Command:
python scripts/train_teacher.py --config configs/teacher_config.yaml
Output:
outputs/teacher_checkpoints/teacher_best.pth

### 2. Generate Pseudo-Labels Using Teacher
Command:
python scripts/generate_pseudolabels.py --teacher_ckpt outputs/teacher_checkpoints/teacher_best.pth --unlabeled_dir data/meta_dataset/unlabeled --save_dir data/meta_dataset/pseudolabels
Output:
data/meta_dataset/pseudolabels/

### 3. Train Student (Semi-Supervised)
Command:
python scripts/train_student.py --config configs/student_config.yaml
Output:
outputs/student_checkpoints/student_best.pth

## What the Student Learns
Because Meta Dataset has 4 total classes, both teacher and student always operate on 4-class segmentation.

If you give the student an image containing 3 or 4 classes:
- It will still output all 4 classes (softmax head has 4 channels).
- Even if some images only contain 1–2 classes, the overall model learns the full class distribution.

## Dataset Requirements
Place dataset as:
data/meta_dataset/
    images/
    masks/
    unlabeled/

Mask labels:
0 = Background
1 = Fire
2 = Smoke
3 = Burned Area

## Config Files
teacher_config.yaml:
- Pure supervised training
- CrossEntropy loss
- Standard augmentations

student_config.yaml:
- Labeled data + pseudo-labeled data
- Consistency regularization
- Weak/strong augmentations

## Final Outputs
After running both stages:
outputs/
    teacher_checkpoints/
        teacher_best.pth
    student_checkpoints/
        student_best.pth
    logs/

The student model becomes the final semi-supervised wildfire segmentation model trained on the Meta Dataset.

## Citation
If you use the Meta Dataset components, please cite this work.

## Contact
For issues, improvements, or experiments, feel free to open an issue or ask for extensions.
