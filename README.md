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
Downloaded and extracted the Oxford-IIIT Pet Dataset.

Step B:

Building DataLoaders (dls)
def is_cat(x): 
    return x[0].isupper()

dls = ImageDataLoaders.from_name_func(
    path, 
    get_image_files(path), 
    valid_pct=0.2, 
    seed=42,
    label_func=is_cat, 
    item_tfms=Resize(224)
)
Step C:

 Training the Model
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(1)

Step D:
Making Predictions & Viewing Results
uploader = SimpleNamespace(data=['path_to_your_image.jpg'])
img = PILImage.create(uploader.data[0])
is_cat, _, probs = learn.predict(img)

print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.4f}")


4. Knowledge Check & Practical Challenge
❓ Q1: What is the role of dls (DataLoaders)?
Answer: dls prepares, batches, and feeds data into the model. It splits data into training and validation sets, applies transformations (like resizing images to 224x224), and maps targets/labels to their respective images.

❓ Q2: Why use a pretrained model like resnet18?
Answer: resnet18 comes pre-equipped with feature-extraction capabilities learned from millions of real-world images. Fine-tuning allows us to achieve high accuracy with significantly smaller datasets and faster training times.

❓ Q3: What does it mean if a model performs well on training data but fails on unseen images?
Answer: This is called Overfitting. The network has memorized the training images instead of generalizing features and patterns.

🛠️ How to Run
Open Google Colab or set up a local PyTorch environment.

Install fast.ai: pip install fastai

Run the notebook .ipynb provided in this repo.
---

### 📂 GitHub par Commit & Push kaise karein:

Terminal / Git Bash mein ye commands chalayein:

```bash
# 1. New repo folder setup
git init
git add README.md
git commit -m "Add fast.ai Lesson 1 documentation and code notes"

# 2. GitHub repo URL link karein aur push karein
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
