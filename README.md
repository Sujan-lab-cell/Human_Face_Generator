# AI Human Face Generation using WGAN-GP

A deep learning project that generates realistic human face images from random noise using a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP). The model is trained on a custom dataset of 2,776 face images and implemented using TensorFlow/Keras on Kaggle GPU.

---

# Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset Description](#dataset-description)
- [Installation](#installation)
- [Requirements](#requirements)
- [Preprocessing and Augmentation](#preprocessing-and-augmentation)
- [Features](#features)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
  - [Generator](#generator)
  - [Critic](#critic)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Output Images](#output-images)
- [Training Loss Graph](#training-loss-graph)
- [Problems Faced](#problems-faced)
- [Solutions Implemented](#solutions-implemented)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

# Overview

Generative Adversarial Networks (GANs) are capable of learning complex data distributions and generating realistic synthetic data.

This project focuses on generating human face images from random noise vectors using a WGAN-GP architecture. The model learns facial structures, hairstyles, backgrounds, clothing patterns, and identity variations directly from a custom face dataset.

The project was developed and trained using TensorFlow/Keras on Kaggle GPU.

---

# Objectives

- Generate realistic human face images.
- Learn facial features from a custom dataset.
- Implement stable GAN training using WGAN-GP.
- Reduce mode collapse and training instability.
- Build a scalable architecture for future upgrades to StyleGAN.

---

# Dataset Description

| Property | Value |
|-----------|---------|
| Dataset Type | Custom Human Face Dataset |
| Total Images | 2,776 |
| Image Format | RGB |
| Resolution | 64 × 64 |
| Faces | Front Facing |
| Background | Mostly Blue |
| Training Platform | Kaggle GPU |

The dataset consists of student face photographs with consistent pose and lighting conditions.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Human-Face-Generation-WGAN-GP.git

cd Human-Face-Generation-WGAN-GP
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python train.py
```

---

# Requirements

```txt
tensorflow
numpy
matplotlib
opencv-python
Pillow
tqdm
scikit-image
```

---

# Preprocessing and Augmentation

## Image Preprocessing

- Resize images to 64×64
- Convert to RGB
- Normalize images to range [-1, 1]

```python
image = (image - 127.5) / 127.5
```

## Data Augmentation

Implemented augmentations:

### Horizontal Flip

```python
RandomFlip("horizontal")
```

### Brightness Adjustment

```python
RandomBrightness(0.05)
```

Benefits:

- Better generalization
- Increased sample diversity
- Reduced overfitting

---

# Features

- Custom Human Face Dataset
- WGAN-GP Training
- Gradient Penalty
- TensorFlow Dataset Pipeline
- GPU Accelerated Training
- Automatic Checkpoint Saving
- Automatic Image Generation
- Loss Tracking
- Data Augmentation
- TensorBoard Logging
- Modular Architecture

---

# Project Structure

```text
Human_Face_Generator/
│
├── dataset/
│
├── generated_images/
│   ├── epoch_1.png
│   ├── epoch_20.png
│   └── ...
│
├── checkpoints/
│
├── graphs/
│   └── loss_curve.png
│
├── models/
│   └── generator.keras
│
├── logs/
│
├── train.py
│
├── notebook.ipynb
│
├── requirements.txt
│
└── README.md
```

---

# Architecture

## Generator

The generator converts random noise vectors into realistic face images.

### Input

```text
100-Dimensional Noise Vector
```

### Layers

```text
Dense
BatchNormalization
LeakyReLU

Conv2DTranspose
BatchNormalization
LeakyReLU

Conv2DTranspose
BatchNormalization
LeakyReLU

Conv2DTranspose
BatchNormalization
LeakyReLU

Conv2DTranspose
Tanh Output
```

### Output

```text
64 × 64 × 3 Face Image
```

---

## Critic

The critic evaluates whether an image is real or generated.

### Layers

```text
Conv2D (64)
LeakyReLU

Conv2D (128)
LeakyReLU

Conv2D (256)
LeakyReLU

Conv2D (512)
LeakyReLU

Flatten

Dense(1)
```

### Notes

- No Sigmoid Activation
- No Dropout
- Designed specifically for WGAN-GP

---

# Training Configuration

```python
IMAGE_SIZE = 64

LATENT_DIM = 100

BATCH_SIZE = 64

GEN_LR = 1e-4

DISC_LR = 1e-4

CRITIC_ITERATIONS = 5

LAMBDA_GP = 10

BETA_1 = 0.0

BETA_2 = 0.9
```

---

# Results

The model successfully learned:

- Face Shape
- Hair Patterns
- Eye Placement
- Facial Symmetry
- Blue Background
- White Shirt Structure
- Lanyard Patterns

Training showed continuous improvement from random noise to recognizable human faces.

---

# Output Images

## Early Training



<img src="generated_images/epoch_20.png" width="300">

## Mid Training

```markdown
![Epoch 50](images/epoch50.png)
```

## Advanced Training

```markdown
![Epoch 75](images/epoch75.png)
```

## Final Output

```markdown
![Epoch 100](images/epoch100.png)
```

---

# Training Loss Graph

Replace with your graph:

```markdown
![Loss Curve](graphs/loss_curve.png)
```

The graph shows balanced learning between the Generator and Critic throughout training.

---

# Problems Faced

## 1. Generator Produced Only Noise

### Observation

Initial outputs contained:

- Random noise
- Blank patches
- Unrecognizable patterns

### Impact

The model failed to learn meaningful facial structures.

---

## 2. Critic Dominated Training

### Observation

Loss values became highly imbalanced:

```text
Generator Loss ≈ -1200

Critic Loss ≈ -6000
```

### Impact

The generator stopped improving.

---

## 3. Training Instability

### Observation

Generated outputs changed dramatically between epochs.

### Impact

Training became unreliable.

---

## 4. Dropout Reduced Critic Performance

### Observation

Dropout layers weakened feature extraction.

### Impact

Poor feedback to the Generator.

---

## 5. Checkpoint Confusion

### Observation

Older checkpoints loaded automatically.

### Impact

Results from different experiments became mixed.

---

## 6. Generated Image Folder Contained Old Outputs

### Observation

Previous epoch images remained in the output folder.

### Impact

Difficult to identify results from current training runs.

---

# Solutions Implemented

## Switched to WGAN-GP

Implemented:

- Wasserstein Loss
- Gradient Penalty

Result:

- Improved stability
- Better convergence

---

## Balanced Learning Rates

Changed:

```python
GEN_LR = 1e-4
DISC_LR = 1e-4
```

Result:

- Balanced Generator and Critic learning

---

## Removed Dropout

Removed all dropout layers from the Critic.

Result:

- Better feature extraction
- Improved image quality

---

## Improved Training Configuration

Used:

```python
CRITIC_ITERATIONS = 5
```

Result:

- Stable updates
- Better generator learning

---

## Reset Experiments Properly

Implemented:

```python
Delete old checkpoints
Delete old generated images
Start training from scratch
```

Result:

- Cleaner experiments
- Reliable comparisons

---

# Future Improvements

- Train on higher-resolution images
- Increase dataset size
- Add FID Score evaluation
- Add Inception Score evaluation
- Upgrade to StyleGAN2
- Build Streamlit Web Application
- Generate 128×128 and 256×256 images
- Deploy trained model online

---

# Author

**Sujan K S**

Artificial Intelligence & Machine Learning

Human Face Generation using WGAN-GP and TensorFlow/Keras
