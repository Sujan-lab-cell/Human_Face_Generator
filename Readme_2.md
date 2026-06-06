# AI Human Face Generation using WGAN-GP

## Table of Contents

* [Overview](#overview)
* [Objectives](#objectives)
* [Dataset Description](#dataset-description)
* [Requirements](#requirements)
* [Installation](#installation)
* [Preprocessing & Augmentation](#preprocessing--augmentation)
* [Project Structure](#project-structure)
* [Model Architecture](#model-architecture)

  * [Generator](#generator)
  * [Critic](#critic)
* [Training Configuration](#training-configuration)
* [Features](#features)
* [Results](#results)
* [Training Evaluation](#training-evaluation)
* [Problems Faced](#problems-faced)
* [Solutions Implemented](#solutions-implemented)
* [Performance Summary](#performance-summary)
* [Future Improvements](#future-improvements)

---

# Overview

This project implements a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) to generate realistic human face images from random noise vectors.

The model was trained on a custom dataset of student face images and improved through multiple experiments involving face cropping, checkpoint management, loss monitoring, and image quality evaluation.

---

# Objectives

* Generate realistic human face images.
* Learn and implement GAN architectures.
* Improve GAN training stability using WGAN-GP.
* Reduce mode collapse and divergence.
* Evaluate generated image quality using Inception Score.
* Build an end-to-end image generation pipeline.

---

# Dataset Description

| Property      | Value               |
| ------------- | ------------------- |
| Dataset Type  | Custom Face Dataset |
| Total Images  | 2,776               |
| Resolution    | 64×64               |
| Channels      | RGB                 |
| Format        | JPG / PNG           |
| Normalization | [-1,1]              |

The dataset contains student face photographs captured under relatively consistent lighting conditions and backgrounds.

---

# Requirements

```bash
tensorflow
keras
numpy
matplotlib
opencv-python
pillow
tqdm
torch
torchmetrics
torch-fidelity
```

---

# Installation

```bash
git clone https://github.com/your-username/human-face-generator.git

cd human-face-generator

pip install -r requirements.txt
```

---

# Preprocessing & Augmentation

## Face Detection and Cropping

A Haar Cascade Face Detector was used to automatically detect and crop faces before training.

Benefits:

* Removes unnecessary background information.
* Forces the GAN to focus on facial features.
* Improves training efficiency.
* Improves generated face quality.

## Image Preprocessing

* Face detection
* Face cropping
* Resize to 64×64
* RGB conversion
* Normalize to [-1,1]

## Data Augmentation

Implemented:

```python
RandomFlip("horizontal")
```

Benefits:

* Increased diversity
* Improved generalization
* Reduced overfitting

---

# Project Structure

```text
Human_Face_Generator/
│
├── dataset/
├── generated_images/
├── checkpoints/
├── graphs/
├── logs/
├── notebook.ipynb
├── README.md
└── requirements.txt
```

---

# Model Architecture

## Generator

Input:

```text
100-Dimensional Noise Vector
```

Architecture:

```text
Dense
↓
BatchNorm
↓
LeakyReLU
↓
Reshape (4×4×512)
↓
Conv2DTranspose 256
↓
BatchNorm
↓
LeakyReLU
↓
Conv2DTranspose 128
↓
BatchNorm
↓
LeakyReLU
↓
Conv2DTranspose 64
↓
BatchNorm
↓
LeakyReLU
↓
Conv2DTranspose 3
↓
Tanh
```

Output:

```text
64×64×3 Face Image
```

---

## Critic

Architecture:

```text
Input Image
↓
Conv2D 64
↓
LeakyReLU
↓
Conv2D 128
↓
LeakyReLU
↓
Conv2D 256
↓
LeakyReLU
↓
Conv2D 512
↓
LeakyReLU
↓
Flatten
↓
Dense(1)
```

Notes:

* No Sigmoid Layer
* No Dropout Layer
* No Batch Normalization

Implemented according to WGAN-GP recommendations.

---
# Architecture Evolution

During development, multiple experiments were performed to improve image quality and training stability.

---

## Experiment 1: Face Cropping + 64×64 Resolution

The dataset was first preprocessed using Haar Cascade face detection to crop and center facial regions before training.

Configuration:

* Face Cropping: Enabled
* Image Resolution: 64×64
* GAN Type: WGAN-GP

Results:

* Generated recognizable faces
* Learned facial structure and background patterns
* Images remained slightly blurry
* Fine details such as eyes and mouth were not consistently generated

Metrics:

```text
Inception Score ≈ 1.41
```

---

## Experiment 2: Face Cropping + 128×128 Resolution

To improve facial detail generation, the training resolution was increased from 64×64 to 128×128 while continuing to use the cropped-face dataset.

### Generator Architecture

```text
100-D Noise Vector
↓
Dense (4×4×512)
↓
BatchNorm + LeakyReLU
↓
Conv2DTranspose (256)
↓
BatchNorm + LeakyReLU
↓
Conv2DTranspose (128)
↓
BatchNorm + LeakyReLU
↓
Conv2DTranspose (64)
↓
BatchNorm + LeakyReLU
↓
Conv2DTranspose (32)
↓
BatchNorm + LeakyReLU
↓
Conv2DTranspose (3)
↓
Tanh
```

Output:

```text
128×128×3 Face Image
```

Results:

* Better facial symmetry
* Improved eye placement
* Improved mouth generation
* More realistic facial features
* Improved overall visual quality

Metrics:

```text
Inception Score ≈ 1.51
```

---

## Comparison

| Feature          | 64×64 Model | 128×128 Model |
| ---------------- | ----------- | ------------- |
| Face Cropping    | Yes         | Yes           |
| Resolution       | 64×64       | 128×128       |
| Inception Score  | ~1.41       | ~1.51         |
| Face Quality     | Good        | Better        |
| Eye Details      | Limited     | Improved      |
| Mouth Details    | Limited     | Improved      |
| Training Time    | Lower       | Higher        |
| GPU Memory Usage | Lower       | Higher        |

---

## Conclusion

The 128×128 model trained on cropped facial images produced the best results.

Key improvements:

* Higher image quality
* Better facial feature generation
* Improved Inception Score
* More realistic face structures

Final Best Result:

```text
Inception Score: 1.5075 ± 0.0952
```

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

# Features

* WGAN-GP Implementation
* Gradient Penalty
* Face Detection & Cropping
* TensorFlow Dataset Pipeline
* Checkpoint Saving
* Generated Image Saving
* TensorBoard Logging
* Loss Monitoring
* Inception Score Evaluation
* GPU Training Support

---

# Results

## Generated Images

### Epoch 20

```html
<img src="generated_images/epoch_20.png" width="300">
```

### Epoch 50

```html
<img src="generated_images/epoch_50.png" width="300">
```

### Epoch 100

```html
<img src="generated_images/epoch_100.png" width="300">
```

### Epoch 133

```html
<img src="generated_images/epoch_133.png" width="300">
```

The model gradually learned:

* Face structure
* Eyes
* Nose
* Mouth
* Hair
* Clothing patterns
* Background consistency

---

# Training Evaluation

## Training Stability

Final observed losses:

```text
Generator Loss ≈ -14
Critic Loss ≈ -26
```

Observations:

* Stable training
* No exploding gradients
* No NaN losses
* No divergence

---

## Mode Collapse Analysis

Checks performed:

* Visual inspection
* Diversity analysis

Observed:

✅ Different face shapes

✅ Different hairstyles

✅ Different genders

✅ Different facial structures

Conclusion:

No significant mode collapse detected.

---

## Divergence Analysis

Results:

```text
Generator NaN : False
Critic NaN    : False
```

Conclusion:

No training divergence detected.

---

## Inception Score Evaluation

Results:

```text
Inception Score : 1.5075
Std             : 0.0952
```

Observation:

Face cropping improved image quality and increased the Inception Score compared to earlier experiments.

---

# Problems Faced

## 1. Generated Images Were Pure Noise

Early training produced blurry noise patterns instead of recognizable faces.

---

## 2. Critic Became Too Strong

Observed:

```text
Generator Loss ≈ -1000
Critic Loss ≈ -6000
```

Result:

* Unstable training
* Poor generator learning

---

## 3. Checkpoint Confusion

Older checkpoints were accidentally reused across experiments.

---

## 4. Mixed Old and New Generated Images

Generated image folders contained outputs from previous runs.

---

## 5. Blurry Eyes and Mouth Regions

Facial details were difficult to learn due to low resolution and background distractions.

---

# Solutions Implemented

## WGAN-GP

Implemented:

* Wasserstein Loss
* Gradient Penalty

Benefits:

* Improved stability
* Reduced mode collapse

---

## Removed Dropout

Removed Dropout layers from the Critic.

Benefits:

* Better feature extraction
* Faster convergence

---

## Face Cropping

Implemented automatic face detection and cropping.

Benefits:

* Better focus on facial features
* Improved image quality
* Higher Inception Score

---

## Fresh Experiment Setup

Before training:

```python
Delete checkpoints
Delete generated images
```

Benefits:

* Reproducible experiments
* Easier debugging

---

## Inception Score Evaluation

Added quantitative evaluation of generated image quality.

Results:

```text
IS = 1.5075 ± 0.0952
```

---

# Performance Summary

| Metric          | Result       |
| --------------- | ------------ |
| Dataset Size    | 2,776 Images |
| Resolution      | 64×64        |
| GAN Type        | WGAN-GP      |
| Face Cropping   | Yes          |
| Training Stable | Yes          |
| Mode Collapse   | No           |
| Divergence      | No           |
| NaN Losses      | No           |
| Inception Score | 1.5075       |
| Checkpointing   | Yes          |
| GPU Training    | Yes          |

---

# Future Improvements

* Train on larger datasets.
* Upgrade to 128×128 resolution.
* Calculate FID Score.
* Implement Spectral Normalization.
* Explore StyleGAN2.
* Improve eye and mouth generation.
* Add latent space interpolation.
* Deploy using Streamlit.

---

# Author

**Sujan K S**

Artificial Intelligence & Machine Learning

Deep Learning • GANs • Computer Vision • Generative AI
