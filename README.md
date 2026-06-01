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
* [Loss Curve](#loss-curve)
* [Problems Faced](#problems-faced)
* [Solutions Implemented](#solutions-implemented)
* [Future Improvements](#future-improvements)

---

# Overview

This project focuses on generating realistic human face images using a custom dataset of 2,776 face photographs. A Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) was implemented using TensorFlow and Keras to improve training stability and image quality.

The model learns the underlying distribution of human faces and generates new synthetic faces from random noise vectors.

---

# Objectives

* Generate realistic human face images from random noise.
* Understand and implement GAN architectures.
* Explore training stability techniques for GANs.
* Reduce mode collapse and unstable training.
* Build a complete end-to-end image generation pipeline.
* Gain hands-on experience with WGAN-GP.

---

# Dataset Description

| Property      | Value               |
| ------------- | ------------------- |
| Dataset Type  | Custom Face Dataset |
| Total Images  | 2,776               |
| Image Size    | 64 × 64             |
| Channels      | RGB                 |
| Format        | JPG / PNG           |
| Normalization | [-1, 1]             |

Dataset contains student face images captured under relatively similar lighting and background conditions.

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
```

---

# Installation

```bash
git clone https://github.com/your-username/human-face-generation.git

cd human-face-generation

pip install -r requirements.txt
```

---

# Preprocessing & Augmentation

## Image Preprocessing

* Resize images to 64×64
* Convert to RGB
* Normalize pixel values to [-1,1]
* Create TensorFlow Dataset Pipeline

## Data Augmentation

Implemented:

```python
RandomFlip("horizontal")
RandomBrightness(0.05)
```

Benefits:

* Increased dataset diversity
* Improved generalization
* Reduced overfitting

---

# Project Structure

```text
Human_Face_Generator/
│
├── dataset/
│
├── generated_images/
│
├── checkpoints/
│
├── graphs/
│
├── models/
│
├── logs/
│
├── human-face-generator.ipynb
│
└── README.md
```

---

# Model Architecture

## Generator

Input:

```text
100-Dimensional Random Noise Vector
```

Architecture:

```text
Dense
↓
Batch Normalization
↓
LeakyReLU
↓
Reshape (4×4×1024)
↓
Conv2DTranspose (512)
↓
Batch Normalization
↓
LeakyReLU
↓
Conv2DTranspose (256)
↓
Batch Normalization
↓
LeakyReLU
↓
Conv2DTranspose (128)
↓
Batch Normalization
↓
LeakyReLU
↓
Conv2DTranspose (3)
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
Conv2D (64)
↓
LeakyReLU
↓
Conv2D (128)
↓
LeakyReLU
↓
Conv2D (256)
↓
LeakyReLU
↓
Conv2D (512)
↓
LeakyReLU
↓
Flatten
↓
Dense(1)
```

Note:

* No Sigmoid Layer
* No Batch Normalization
* No Dropout

This design follows WGAN-GP recommendations.

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

# Features

* Custom WGAN-GP Implementation
* Gradient Penalty
* TensorFlow Dataset Pipeline
* Automatic Checkpoint Saving
* Generated Image Saving
* Loss Visualization
* GPU Training Support
* Custom Face Dataset
* Data Augmentation
* Training Monitoring

---

# Results

## Generated Images

### Epoch 20
<p align="center">
  <img src="generated_images/epoch_20.png" width="300">
  <br>
  <b>Epoch 20 Output</b>
</p>

### Epoch 50

![Epoch 50](images/epoch50.png)

### Epoch 75

![Epoch 75](images/epoch75.png)

### Epoch 100

![Epoch 100](images/epoch100.png)

Generated images gradually evolved from noise to recognizable face structures with facial features, hair, background, and clothing patterns.

---

# Loss Curve

![Loss Curve](graphs/loss_curve.png)

The Generator and Critic losses gradually stabilized, indicating balanced adversarial training and improved convergence.

---

# Problems Faced

## 1. Generated Images Were Pure Noise

Initial training produced random noise and blurry patches instead of faces.

### Impact

* No recognizable facial features
* Poor convergence

---

## 2. Critic Dominated Training

Observed loss values:

```text
Generator Loss ≈ -1000
Critic Loss ≈ -6000
```

### Impact

* Generator stopped improving
* Unstable training

---

## 3. Dropout Reduced Critic Performance

Dropout layers were originally included in the Critic.

### Impact

* Weaker feature extraction
* Slower convergence

---

## 4. Checkpoint Conflicts

Old checkpoints were automatically loaded during new experiments.

### Impact

* Inconsistent results
* Difficult debugging

---

## 5. Experiment Tracking Issues

Generated image folders contained outputs from previous runs.

### Impact

* Difficult to compare experiments
* Confusing results

---

# Solutions Implemented

## WGAN-GP

Replaced standard GAN loss with:

* Wasserstein Loss
* Gradient Penalty

Benefits:

* Stable training
* Better convergence
* Reduced mode collapse

---

## Balanced Learning Rates

Changed training configuration to:

```python
GEN_LR = 1e-4
DISC_LR = 1e-4
CRITIC_ITERATIONS = 5
```

Benefits:

* Better Generator-Critic balance
* Improved image quality

---

## Removed Dropout

Removed all Dropout layers from the Critic.

Benefits:

* Better feature learning
* Faster convergence

---

## Fresh Training Runs

Before training:

```python
Delete old checkpoints
Delete old generated images
```

Benefits:

* Reproducible experiments
* Easier debugging

---

## Dataset Verification

Verified:

```python
images.min() = -1.0
images.max() = 1.0
```

Confirmed preprocessing pipeline was correct.

---

# Future Improvements

* Train on larger datasets
* Generate higher-resolution faces
* Implement Spectral Normalization
* Calculate FID Score
* Calculate Inception Score
* Upgrade to StyleGAN2
* Deploy using Streamlit
* Add latent space interpolation
* Support 128×128 and 256×256 image generation

---

# Author

**Sujan K S**

Artificial Intelligence & Machine Learning

GANs • Deep Learning • Computer Vision • Generative AI
