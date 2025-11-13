# 🥔 Potato Leaf Disease Classification using CNN

## 📝 **Project Overview**

This project focuses on developing a **Convolutional Neural Network (CNN)** model capable of classifying potato leaf diseases into three categories:

* **Early Blight**
* **Late Blight**
* **Healthy**

The goal is to enable **early detection** of crop diseases, helping farmers and agricultural systems reduce crop loss and improve yield.

---

## 📂 **Dataset**

The dataset was sourced from Kaggle: **emmarex/plantdisease**.
It includes potato leaf images categorized into:

* `Potato___Early_blight`
* `Potato___Late_blight`
* `Potato___healthy`

The dataset was downloaded, extracted, and prepared for training.

---

## 🔧 **Preprocessing & Data Augmentation**

### **📌 Data Splitting**

The dataset was split into:

* **80%** Training
* **10%** Validation
* **10%** Test

### **📌 Image Preprocessing**

* All images resized to **256 × 256 pixels**
* Pixel values rescaled to **[0, 1]**

### **📌 Data Augmentation**

To improve generalization and reduce overfitting, the following augmentations were applied:

* Random horizontal & vertical flips
* Random rotations
* Random zoom operations
* Brightness & contrast adjustments
* Random translations
* Gaussian noise injection

---

## 🧠 **Model Architecture (CNN)**

A Sequential model was built using **TensorFlow/Keras**, consisting of:

* **Resizing** layer (256×256)
* **Rescaling** layer ([0,1])
* Multiple **Conv2D layers** (32/64 filters, 3×3 kernel, ReLU activation)
* **MaxPooling2D layers** for downsampling
* **Flatten** layer
* **Dense layer** (64 units, ReLU activation)
* **Output Dense layer** (3 units, softmax activation)

This architecture extracts hierarchical image features and performs **multi-class classification**.

---

## 🎯 **Training Configuration**

* **Optimizer:** Adam
* **Loss:** SparseCategoricalCrossentropy
* **Metrics:** Accuracy
* **Batch Size:** 32
* **Epochs:** Max 30
* **Callbacks:** EarlyStopping

  * Patience: 10
  * Monitors `val_loss`
  * Restores best weights

---

## 📊 **Results**

| Metric            | Value        |
| ----------------- | ------------ |
| **Test Accuracy** | **≈ 99.06%** |
| **Test Loss**     | **≈ 0.023**  |

* A **confusion matrix** confirms strong performance across all three classes.
* The model demonstrates excellent generalization and robustness.

---

## 🌐 **Deployment (Gradio Interface)**

To make the model accessible, a simple interactive interface was created using **Gradio**.

### ✔️ Features:

* Upload a potato leaf image
* Receive a prediction with confidence scores
* Supports all three classes (Early Blight, Late Blight, Healthy)

Gradio enables easy testing without needing to run Jupyter notebooks.

---

## 🧰 **Technologies Used**

* Python
* TensorFlow / Keras
* NumPy, Pandas
* Matplotlib, Seaborn
* Gradio
* Google Colab / Jupyter Notebook

---
