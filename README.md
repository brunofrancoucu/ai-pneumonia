# Pneumonia Detection

![Output](/docs/thumbnail.jpg)

A PyTorch-based deep learning pipeline for **detecting pneumonia in chest X-rays**, utilizing contrastive pre-training (ResNet-50) and supervised linear evaluation. The repository features a complete workflow from offline image standardization to Grad-CAM heatmap explainability.

## Installation

```bash
uv sync
```

> CPU/GPU [config file](/config.json)

### Running

```bash
uv run python src/train.py  # Training (optiona)
uv run python src/test.py   # Testing (Weights at src/weights)
```

### Validation Prediction

Generate `.csv` fromatted prediction for validation dataset from kaggle

```bash
uv run python src/validation.py
```
> Output: `src/prediction.csv`

### Generate Heatmap

Generate a heatmap version of the x-ray scan.

```bash
uv run python src/heatmap.py .\src\heatmap.py src/dataset/test/pneumo_1465v2537.jpeg
```
> Output: `src/prediction.csv`

# Training Pipeline

![ResNet 50](/docs/resnet50.JPG)

## 1. Standardization (offline)

### Objective: 
Minimize the **variance** caused by irrelevant **noise**. For instance, darker/brighter X-ray machine outputs or extra pixels not within lungs area.

### Result:
Clean, uniform dataset. Network's loss function only penalizes mistakes made on the actual pathology.

Also allows for lower image size and increased VRAM effectiveness (higher batch size)

> **Requires**: Once deployed, images must go through standardization.

### Process:
-   Enforce CLAHE (Illumination & Contrast)
-   Aspect Ratio
-   Cropping (not lungs)

## 2. Augmentation

![Augmentation](/docs/augmentation.JPG)

> Reference. [src/model/augmentation.py](/src/model/augmentation.py)

### Objective:
Avoid overfitting of characteristics (lighting | background | rotation), generate positive pairs $(i, j)$. Matching "real world" plausible variation, example for rotation: [test_0055.jpeg](/assets/test/test_0055.jpeg)

### Process:

- Translation (5% limit)
- Scaling, min-scale of 0.75
- Rotation (max 15 deg)
- Horizontal Flip 

> <small>Consideration: Moves the heart to the right side of the chest (simulating situs inversus). For pneumonia detection—which focuses on lung tissue opacities—this anatomical mirroring is completely acceptable and highly beneficial for training. It would only be problematic if you were specifically diagnosing cardiomegaly (enlarged heart) based on lateral orientation.</small>

## 3. Encoder (Contrastive Pre-Training, phase 1)

The backbone (detective), extract & summarize relevant biological features. Neural network, Pytorch architecture, and two-stage training process required in contrastive learning?.

![Phase 1](/docs/phase_1.JPG)

> Reference. [src/train.py](/src/train.py)

### Objective:
Compress pixels into a hidden feature vector. Extract a high dimensional vector representation ($h$). Map to lower dimension latent space ($z$)

### Result:
Force the latent codes of the two augmented crops to be nearly identical.

### Process:
Phase 1 Contrastive pre-training:
-   Convolutional Network (**ResNet-50** or DenseNet)
-   MLP: Projection Head: Maps high-dimensional h to lower-dimensional z (high detailed and overfitted will be deleted!)

#### Optimizer: **AdamW**

<!-- > Reference. [src/utils/loss.py](/src/model/utils/scheduler.py) -->

#### Scheduler: Cosine

![Scheduler](/docs/scheduler.JPG)

> Reference. [src/model/utils/scheduler.py](/src/model/utils/scheduler.py)

#### Loss: **SupCon**
<!-- TODO: Update to supcon loss & move to proper stage -->

positive pair of augmented views $(i, j)$ generated from the same underlying scan, the network optimizes statistical expectations by minimizing this loss:

$$L_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$

Where $\text{sim}(\cdot, \cdot)$ is the cosine similarity between the two feature vectors, and $\tau$ is a temperature hyperparameter that scales the penalty for hard negative examples within your training batch.

> Reference. [src/model/utils/loss.py](/src/model/utils/loss.py)

## 4. Classification Head (Linear Evaluation)

![Phase 2](/docs/phase_2.JPG)

> Reference. [src/train.py](/src/train.py)

### Process:
-   Discard projection head
-   Attach class head to backbone encoder

fine tuning loop
-   freezed weights, standard Cross-Entropy Loss teach difference 

## 5. Fine-Tuning

Full Network Fine-Tuning, both Backbone and Classification train together

![Phase 3](/docs/phase_3.JPG)

> Reference. [src/train.py](/src/train.py)

## 6. Heatmap
Using a post-training explainability technique called **Grad-CAM** (Gradient-weighted Class Activation Mapping).

### Result:
Highlight exactly which spatial regions of the X-ray caused the network to make its decision.