# 🧠 Deep Learning Journey: Part 1 — How AI Models Learn

Welcome to my Deep Learning study repository! This log documents my theoretical understandings, code implementations, and project milestones.

---

## 📌 Table of Contents
1. [Core Philosophy: Rule-Based vs Machine Learning](#-1-core-philosophy-rule-based-vs-machine-learning)
2. [The Deep Learning Training Loop](#-2-the-deep-learning-training-loop)
3. [Transfer Learning & Architecture (ResNet-18)](#-3-transfer-learning--architecture-resnet-18)
4. [Knowledge Check & Concepts](#-4-knowledge-check--concepts)

---

## 💡 1. Core Philosophy: Rule-Based vs Machine Learning

* **Traditional Programming:**
$$\text{Data} + \text{Rules} \longrightarrow \text{Answers}$$
* **Machine Learning:**
$$\text{Data} + \text{Answers} \longrightarrow \text{Rules}$$

Instead of manually writing rules for recognizing features (whiskers, fur, ears), we feed thousands of labeled images into a Neural Network to let it learn patterns directly.

---

## 🔄 2. The Deep Learning Training Loop

At its core, model training relies on an iterative feedback optimization process:

1. **Forward Pass:** Input images pass through network layers to calculate class probabilities.
2. **Loss Function:** Quantifies the divergence between predicted output and true target (e.g., Cross-Entropy Loss).
3. **Backpropagation:** Computes gradients of the loss function using the Chain Rule.
4. **Optimization:** Updates network weights (e.g., Adam / SGD) to minimize loss value.

---

## ⚡ 3. Transfer Learning & ResNet-18

Instead of training a model from scratch, we leverage **Transfer Learning**:
* **Pretrained Weights:** `resnet18` comes pre-trained on ImageNet (1M+ images), already understanding edge, shape, and texture representations.
* **Fine-Tuning:** We adapt the final layer specifically to our target dataset.

---

## ❓ 4. Knowledge Check & Concepts

<details>
<summary><b>1. What is the role of <code>dls</code> (DataLoaders)?</b></summary>

> `dls` organizes raw images and labels into mini-batches, manages data split (training vs validation), and feeds normalized batches to the GPU.
</details>

<details>
<summary><b>2. Why use a pretrained model?</b></summary>

> Saves huge training time and compute resources while yielding higher accuracy with smaller datasets.
</details>

<details>
<summary><b>3. What happens if training score is high but validation performance is poor?</b></summary>

> This indicates **Overfitting** (the model memorized the training set). Addressed via Data Augmentation, Regularization (Dropout, Weight Decay), or Early Stopping.
</details>

---

## 📍 Progress Tracker
- [x] Part 1: How AI Models Learn
- [ ] Part 2: Fast.ai Training Loop & Metrics
- [ ] Part 3: Neural Networks from Scratch
