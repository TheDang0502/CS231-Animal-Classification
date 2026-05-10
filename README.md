# 🐾 Animal Image Classifier

An animal image classification project using both **HOG traditional computer vision techniques** and **VGG16 deep learning feature extraction methods** combined with a **Random Forest classifier**.

Developed for the **CS231 – Computer Vision** course.

---

## 📋 Project Overview

This project investigates the effectiveness of different feature extraction methods for **multi-class animal image classification**.

| Method | Type |
|----------|------------------------------|
| **HOG** | Handcrafted feature extraction |
| **VGG16** | CNN deep feature extraction |

Both methods are evaluated using a:

- **Random Forest Classifier**

### 🏆 Best Performing Method

> **VGG16 + PCA + Random Forest**

This combination achieved the highest performance and is used in the final demo application.

---

## 👥 Team Member

| Name | Student ID |
|------|------------|
| **Trần Thế Đăng** | **23520238** |

---

## 🐶 Supported Animal Classes

`butterfly` • `cat` • `chicken` • `cow` • `dog`  
`elephant` • `horse` • `sheep` • `spider` • `squirrel`

---

## 🏗️ Model Pipeline (Best Method)

```text
Input Image
      ↓
Resize to 224×224
      ↓
VGG16 Feature Extraction
(fc2 → 4096 features)
      ↓
StandardScaler
      ↓
PCA Dimensionality Reduction
      ↓
Random Forest Classifier
      ↓
Predicted Animal Class
```

---

## 📁 Project Structure

```text
project/
│── app.py                      # Gradio web application
│── README.md
│── modules/                    # Saved model files
│   ├── hog_scaler.pkl
│   ├── hog_pca.pkl
│   ├── hog_random_forest_model.pkl
│   ├── vgg16_scaler.pkl
│   ├── vgg16_pca.pkl
│   └── vgg16_random_forest_model.pkl
│
│── notebooks/
│   ├── CS231_HOG.ipynb
│   ├── CS231_VGG16.ipynb
│   └── demo_prediction.ipynb
│
│── src/
│   └── predict.py              # Prediction pipeline
```

---

## 🚀 Getting Started

### Requirements

* Python >= 3.13
* [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd proj

# Install dependencies
uv sync
```

### Usage

#### 1. Run Web Application (Recommended)

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

Features:

* **Single Prediction**: Upload an image
* **Animal Classification**: Predict one of 10 supported animal classes
* **Confidence Score**: View prediction probabilities

#### 2. Command Line

```bash
# Predict image
python src/predict.py path/to/image.jpg

# Interactive mode
python src/predict.py
```

#### 3. Jupyter Notebook

```bash
# Run HOG experiment
jupyter notebook notebooks/CS231_HOG.ipynb

# Run VGG16 experiment
jupyter notebook notebooks/CS231_VGG16.ipynb

# Demo predictions
jupyter notebook notebooks/demo_prediction.ipynb
```

#### 4. Python API

```python
from src.predict import AnimalClassifier

# Initialize classifier
classifier = AnimalClassifier('modules')

# Make prediction
result = classifier.predict('path/to/image.jpg')

print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 📊 Experimental Results

| Method                    | Accuracy   |
| ------------------------- | ---------- |
| **HOG + Random Forest**   | **36.08%** |
| **VGG16 + Random Forest** | **96.05%** |

The experimental results show that **VGG16 significantly outperforms HOG**, demonstrating the effectiveness of deep feature extraction for animal image classification.

---

## 🔧 Dependencies

| Package        | Version  |
| -------------- | -------- |
| tensorflow-cpu | >=2.20.0 |
| scikit-learn   | ==1.6.1  |
| opencv-python  | >=4.11.0 |
| numpy          | >=2.3.5  |
| gradio         | >=6.0.2  |
| matplotlib     | >=3.10.7 |

---

## 📝 Notes

* First prediction may take longer as models are loaded into memory
* Uses `tensorflow-cpu` by default
* For GPU support, replace with `tensorflow`
* `scikit-learn==1.6.1` is required to match saved model files
* The final demo application uses **VGG16 + PCA + Random Forest**

---

## 📚 Course Information

**Course**: CS231 – Computer Vision
**University**: University of Information Technology (UIT)

---

## 🙏 Acknowledgments

* [VGG16](https://arxiv.org/abs/1409.1556) — Visual Geometry Group, Oxford
* [Animals-10 Dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10) — Kaggle
* [Gradio](https://gradio.app/) — Web UI framework

```
```