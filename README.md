# AI Human Face Generation using WGAN-GP

## Project Overview

This project aims to generate realistic human face images using a custom dataset of 2,776 face images. The model is implemented using TensorFlow/Keras and trained on Kaggle GPU using a WGAN-GP (Wasserstein GAN with Gradient Penalty) architecture.

---

## Current Progress

### Dataset
- Custom face dataset
- Total Images: 2,776
- Resized to 64×64
- RGB Images
- Normalized to [-1, 1]

### Data Pipeline
- TensorFlow Dataset API
- Shuffle
- Batch Processing
- Cache
- Prefetch
- Data Augmentation:
  - Horizontal Flip
  - Small Brightness Adjustment

### Generator Architecture
- Input Noise Vector: 100
- Dense Layer
- Batch Normalization
- LeakyReLU
- Conv2DTranspose Layers
- Tanh Output Layer
- Output Size: 64×64×3

### Critic Architecture
- Conv2D Layers
- LeakyReLU Activations
- Flatten Layer
- Dense Output Layer
- Dropout Removed for Better Stability

### Training Setup
- WGAN-GP Loss
- Gradient Penalty
- Adam Optimizer
- Generator LR = 1e-4
- Critic LR = 1e-4
- Critic Iterations = 5
- Lambda GP = 10

---

# Problems Faced & Solutions

## 1. Generated Images Were Pure Noise

### Problem
Initial training produced only random noise and blurry patches instead of faces.

### Cause
Training instability and imbalance between Generator and Critic.

### Solution
- Switched to WGAN-GP
- Added Gradient Penalty
- Tuned Learning Rates
- Removed unnecessary layers

---

## 2. Critic Became Too Strong

### Problem

Training logs showed:

```text
Generator Loss ≈ -1000
Critic Loss ≈ -6000
