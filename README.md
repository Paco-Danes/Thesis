# 📡 A Robust Person Identification Method Through Wi-Fi Signals

![Wi-Fi Person ID Diagram](https://img.shields.io/badge/WiFi-Sensing-blue)
![Deep Learning](https://img.shields.io/badge/DeepLearning-CNN-brightgreen)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

## 📖 Overview

This repository contains the code, dataset structure, and documentation for my master's thesis project, developed at **Sapienza University of Rome**. The thesis proposes a novel, non-intrusive person identification system that leverages **Wi-Fi Channel State Information (CSI)** and deep learning—specifically, **Convolutional Neural Networks (CNNs)**—to recognize individuals based on the unique disturbances their body causes to wireless signals.

> 🧠 “From electromagnetic waves to identity recognition—this work shifts the paradigm from camera-based surveillance to privacy-conscious Wi-Fi sensing.”

---

## 🔬 Research Motivation

Traditional person identification relies heavily on vision-based systems like cameras. However, these systems suffer from challenges including occlusions, lighting variations, and **privacy concerns**. This thesis explores the use of **commodity Wi-Fi hardware** to build a secure and effective identification system based on **CSI data**, with benefits such as:

- ⚡ Non-intrusive operation  
- 🛡️ Enhanced privacy  
- 📶 Pervasive infrastructure usage  

---

## 🧪 Methodology

### 📥 Data Acquisition

- **Devices:** ESP32 microcontrollers  
- **Participants:** 6 individuals  
- **Samples:** 90,000+ CSI packets per person  
- **Scenario:** Controlled indoor space with static positions and six orientations  

### ⚙️ Preprocessing

- **Amplitude Outlier Removal:** Hampel Filter + Wavelet Denoising  
- **Phase Sanitization:** Linear transformation to correct time offset and noise  
- **Image Construction:** Combination of amplitude and phase into 200×104 matrix per sample  

### 🧠 Model

A custom **CNN** model processes the CSI-derived images with the following features:

- Strided convolutions aligned to alternating amplitude/phase input  
- Spatial dropout for regularization  
- Max-pooling and ReLU activations  
- Fully connected layers with softmax output  

📈 **Accuracy** achieved:

| # of Classes | Accuracy |
|--------------|----------|
| 2            | 94.2%    |
| 3            | 93.9%    |
| 4            | 93.3%    |
| 5            | 92.9%    |
| 6            | 92.4%    |

---

## 📊 Results

- The model performs with **>92% accuracy** even for 6-class classification  
- **Cross-validation** and **data augmentation** ensure robust performance  
- Results suggest strong correlation between individual body characteristics and CSI signal deformation  

📄 Refer to the thesis PDF for more:

- Page 22: Complete pipeline overview  
- Page 24: CNN architecture  
- Page 28: Validation loss/accuracy trends  

---

## 🗂️ Repository Structure

