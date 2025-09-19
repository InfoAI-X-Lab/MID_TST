# MID_TST
# Dual-Stage Training (with MINE Model) - CIFAR-10 Version

## 📌 Project Overview

This project implements a **two-stage training pipeline** combined with the **Mutual Information Neural Estimator (MINE)** model, which introduces mutual information constraints during the feature extraction stage to improve the robustness of the student model.
This version uses the **CIFAR-10** dataset, a ResNet-based model, and provides pre-trained MINE models for quick verification of the training process.

---

## 📂 Project Structure

```
.
├── models/
│   ├── Mine.py               # MINE model definition
│   ├── resnet18.py               # MINE model definition
│   ├── LeNet5.py               # MINE model definition
├── train.py                  # Main training script (Stage 1 / Stage 2)
└── utils/
    ├── add_noise_to_model.py # Gaussian noise injection utility
    ├── losses.py             # compute_stage1_loss / compute_stage2_loss
```

---

## 🔧 Requirements

Install dependencies:

```bash
pip install torch torchvision tqdm
```

> If using GPU, please install PyTorch and torchvision matching your CUDA version.

---

## 📦 Dataset

The code automatically uses **CIFAR-10**:

* 10 classes, 32×32 color images
* Automatically downloaded to `./data` on first run
* Uses the train/test split provided by `torchvision.datasets.CIFAR10`

---

## 🚀 Quick Start

### 1. Prepare the MINE Models

Before running Stage 1, you need to train and save two MINE models:

* `mine.pth`: trained on clean features
* `mine_nm.pth`: trained on noisy features

In the configuration file:

```python
cfg.train.mine_path = "mine.pth"
cfg.train.mine_nm_path = "mine_nm.pth"
```

### 2. Prepare the Teacher Model

In the configuration file:

```python
cfg.train.teacher_path = "teacher_model.pth"
```

This model will be used for distillation in Stage 1.

### 3. Run Stage 1

```bash
python train.py 
Stage = stage 1
```

Process:

* Load weights from the teacher model
* Load MINE models from `mine.pth` and `mine_nm.pth`
* Inject Gaussian noise into the student model
* Use `compute_stage1_loss` to calculate loss (classification + distillation + mutual information + Lipschitz regularization)
* Save the student feature extractor and classifier weights to `cfg.train.save_dir`

### 4. Run Stage 2

```bash
python train.py 
Stage = stage 2
```

Process:

* Load the student model from Stage 1 results
* Freeze the feature extractor and train only the classifier
* Optionally use the teacher model for distillation
* Use `compute_stage2_loss` to calculate loss (classification + distillation + Lipschitz regularization)
* Save the updated classifier weights to `cfg.train.save_dir`

---

## ⚙️ Core Modules

### 1. MINE Model

File: `models/Mine.py`

* Estimates mutual information between clean and noisy features
* Two instances loaded in the training script:

  * `mine`: clean features
  * `mine_nm`: noisy features

### 2. Noise Injection

Function: `noisy_weights()`

* Temporarily adds Gaussian noise to model parameters during forward pass
* Automatically restores parameters after exiting the context

### 3. Loss Functions

File: `utils/losses.py`

* `compute_stage1_loss`: classification loss + distillation loss + MI loss + Lipschitz regularization
* `compute_stage2_loss`: classification loss + distillation loss + Lipschitz regularization

---

## 📝 Notes

* Teacher model and both MINE model weights must be prepared before running Stage 1.
* CIFAR-10 will be downloaded automatically on first run.
* Only the dataset loading part is replaced with CIFAR-10; all other training logic remains unchanged.
* Ensure `cfg.train.save_dir` exists or can be created.

---

## 📜 License

This project is for research and educational purposes only.
