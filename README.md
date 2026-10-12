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


