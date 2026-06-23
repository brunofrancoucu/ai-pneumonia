# Pipeline
___

## 1. Standarization (offline)

### Objective: 
Minimize the **variance** caused by irrelevant **noise**. For instance, darker/brighter X-ray machine outputs or extra pixels not within lungs area.

### Result:
Clean, uniform dataset. Network's loss function only penalizes mistakes made on the actual pathology.

> **Requires**: Once deployed, images must go through standarization.

### Process:
-   Enforce CLAHE (Illumination & Contrast)
-   Aspect Ratio
-   Cropping (not lungs)

## 2. Augmentation

### Objective:
Avoid overfitting of characteristics (lighting | background | rotation), generate positive pairs $(i, j)$. Matching "real world" plausible variation, example for rotation: [test_0055.jpeg](/assets/test/test_0055.jpeg)

### Process:

- Translation (5% limit)
- Horizontal Flip 

> <small>Consideration: Moves the heart to the right side of the chest (simulating situs inversus). For pneumonia detection—which focuses on lung tissue opacities—this anatomical mirroring is completely acceptable and highly beneficial for training. It would only be problematic if you were specifically diagnosing cardiomegaly (enlarged heart) based on lateral orientation.</small>

- Scaling, min-scale of 0.75
- Rotation (max 15 deg)

## 3. Encoder

Neural network, Pytorch architecture, and two-stage training process required in contrastive learning?.

### Objective:
Compress pixels into a hidden feature vector. Extract a high dimensional vector representation ($h$). Map to lower dimension latent space ($z$)

### Result:
Force the latent codes of the two augmented crops to be nearly identical.

### Process:
Phase 1 Contrastive pre-training:
-   Convolutional Network (**ResNet-50** or DenseNet)
-   Pass small multi-layer perceptron (MLP)

Phase 2 Classification (linear evaluation):
-   Discard projection head
-   Attach class head to backbone encoder

fine tuning loop
-   freezed weights, standard Cross-Entropy Loss teach difference 

### Loss Function

positive pair of augmented views $(i, j)$ generated from the same underlying scan, the network optimizes statistical expectations by minimizing this loss:

$$L_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$

Where $\text{sim}(\cdot, \cdot)$ is the cosine similarity between the two feature vectors, and $\tau$ is a temperature hyperparameter that scales the penalty for hard negative examples within your training batch.

### Projection Head?
lower dimension z

## 4. Heatmap
Using a post-training explainability technique called **Grad-CAM** (Gradient-weighted Class Activation Mapping).

### Result:
Highlight exactly which spatial regions of the X-ray caused the network to make its decision.