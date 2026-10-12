# 🚀 Fast.ai Lesson 1 — Deep Learning Foundations & Image Classifier

Welcome to my **fast.ai Journey**! This repository covers **Lesson 1 (Part 1)** of fast.ai, focusing on Deep Learning fundamentals, Transfer Learning, and building an image classifier using PyTorch and fast.ai.

---

## 📌 1. How Does an AI Model Learn?

Instead of writing manual rules for every edge case, a Neural Network learns patterns directly from labeled data during training.

$$\text{Input} \longrightarrow \text{Model} \longrightarrow \text{Prediction} \longrightarrow \text{Evaluation} \longrightarrow \text{Improvement}$$

* **Input:** Image (e.g., Cat or Dog)
* **Target:** Label (`cat` or `dog`)
* **Model:** Neural Network learning visual patterns
* **Output:** Predicted category

---

## 🔄 2. Transfer Learning & Pretrained Models

Training a deep learning network from scratch requires huge compute and millions of images. We use **Transfer Learning** via a pretrained architecture like **ResNet-18**.

* **Pretrained ResNet-18:** Pre-learned feature representations (edges, gradients, textures) trained on ImageNet.
* **Fine-Tuning:** Adapting these pre-learned weights to classify our specific dataset (Oxford-IIIT Pet Dataset).

---

## 💻 3. Code Breakdown & Implementation

### Step A: Setup & Dataset Download
```python
from fastai.vision.all import *

path = untar_data(URLs.PETS)/'images'
