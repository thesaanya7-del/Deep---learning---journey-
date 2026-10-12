# 🧠 Deep Learning Log: From Fundamentals to Applications

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/fast.ai-0078D4?style=for-the-badge&logo=fastai&logoColor=white" />
<img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=525252" />
</p>

Welcome to my Deep Learning study journal! This repository tracks my hands-on learning, practical exercises, and theoretical concepts as I master Deep Learning.

---

## 📚 Table of Contents
- [Part 1: How AI Models Learn](#-part-1-how-ai-models-learn)
- [Transfer Learning & Pretrained Networks](#-transfer-learning--pretrained-networks)
- [Fast.ai Implementation](#-fastai-implementation)
- [Knowledge Check & Q/A](#-knowledge-check--qa)
- [Roadmap](#-learning-roadmap)

---

## 📌 Part 1: How AI Models Learn

Instead of hardcoding rules manually, neural networks learn patterns directly from labeled training data.

### 🔄 The Fundamental Learning Loop

$$\text{Input Data} \longrightarrow \text{Model} \longrightarrow \text{Prediction} \longrightarrow \text{Evaluation} \longrightarrow \text{Improvement}$$

| Component | Description | Example |
| :--- | :--- | :--- |
| **Input** | Raw visual data fed into the network | Images of cats and dogs |
| **Target** | Ground truth labels | `cat` or `dog` |
| **Model** | Architecture learning visual patterns | Convolutional Neural Network |
| **Prediction**| Output probability class | `cat` (98% confidence) |

---

## ⚡ Transfer Learning & Pretrained Networks

Training deep networks from scratch requires massive computation and huge datasets. We use **Transfer Learning** to build efficient models faster.

> 💡 **What is a Pretrained Model?**
> A model like **ResNet-18** has already learned basic visual features (edges, shapes, textures) on millions of images from ImageNet. We only **fine-tune** its final layers for our specific task.
## 🛠️ Fast.ai Implementation

### 1. Data Pipeline (`DataLoaders`)
`DataLoaders` (`dls`) handle raw files, map labels, perform train/validation splits, and push batched data to the GPU.

```python
from fastai.vision.all import *

# Download dataset & build DataLoaders
path = untar_data(URLs.PETS)/'images'
dls = ImageDataLoaders.from_name_func(
path, get_image_files(path), valid_pct=0.2, seed=42,
label_func=is_cat, item_tfms=Resize(224)
)
# Initialize learner with pre-trained ResNet-18
learn = vision_learner(dls, resnet18, metrics=error_rate)

# Fine-tune model for 1 epoch
learn.fine_tune(1)
 Knowledge Check & Q/A

<details>

<summary><b>❓ 1. What is the role of <code>dls</code> (DataLoaders)?</b></summary>





<code>dls</code> organizes raw image files and labels into structured minibatches, handles validation splits (e.g., 20%), and efficiently streams batches to the GPU/CPU during training.

</details>

<details>

<summary><b>❓ 2. Why use a pretrained model like ResNet-18?</b></summary>





It saves compute time and requires significantly less data. ResNet-18 already understands fundamental visual features, so we only need to adapt its final layer for our target dataset.

</details>

<details>

<summary><b>❓ 3. What if performance is great on training data but poor on new images?</b></summary>





This indicates <b>Overfitting</b>. The model memorized specific details of the training set instead of learning generalizable features.

</details>

🗺️ Learning Roadmap

 [x] Part 1: Fundamentals, Data Pipelines, & Fine-Tuning Pretrained Models

 [ ] Part 2: Detailed fast.ai Workflow, Learning Rate Finder, & Loss Functions

 [ ] Part 3: Image Augmentations & Improving Accuracy

 [ ] Part 4: Deploying Model as a Web Application



