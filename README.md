# 🕺🤖 Real-Time Body Language Classifier

A **production-grade machine learning application** that uses **Computer Vision** and **Classification Algorithms** to detect and categorize **human body language and emotions in real time** via webcam input.

---

## 🌟 Overview

This project implements a **complete end-to-end machine learning pipeline** using **MediaPipe Holistic** for landmark extraction and **Scikit-learn** for real-time emotion classification.

The system analyzes **543 body landmarks** (Face, Pose, Hands) per frame and classifies the following emotional states:

- 😀 **Happy** – smiling, open posture  
- 😔 **Sad** – slumped shoulders, frowning  
- 🏆 **Victorious** – arms raised, V-signs  
- 😠 **Angry** – tense posture, furrowed brow  
- 😐 **Normal** – neutral baseline state  

---

## 🛠️ Technical Stack

- **Computer Vision:** MediaPipe (Holistic API)
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Real-Time Interface:** OpenCV
- **Evaluation & Visualization:** Seaborn, Matplotlib

---

## 📈 Performance & Evaluation

- **Evaluation Method:** 10-Fold Cross-Validation  
- **Best Model:** Logistic Regression (with StandardScaler pipeline)  
- **Average Accuracy:** ~96%  

### Metrics Summary
- Precision & Recall: Balanced across all emotion classes
- Confusion Matrix: High diagonal dominance with minimal class confusion  
  (e.g., *Sad vs Normal*)

---

## 🚀 How It Works

### 1️⃣ Feature Extraction
Using **MediaPipe Holistic**, each frame generates **1,629 features**:

| Component | Landmarks |
|---------|-----------|
| Face Mesh | 468 |
| Pose | 33 |
| Hands | 21 per hand |
| **Total Points** | **543 landmarks** |
| **Total Features** | **1,629 (x, y, z, visibility)** |

---

### 2️⃣ Model Training
- Landmarks are flattened and stored in CSV format
- A **Scikit-learn pipeline** is used:
  - `StandardScaler` → normalizes landmark coordinates
  - `LogisticRegression` → emotion classification
- Normalization ensures camera distance does not affect predictions

---

### 3️⃣ Real-Time Inference
- Trained model (`.pkl`) is loaded into an OpenCV webcam loop
- Each frame:
  - Extracts landmarks
  - Performs prediction
  - Displays emotion label + confidence score in real time

---

## 📂 Project Structure

```bash
├── coords.csv                 # Raw landmark dataset
├── Body_Language_Decoder.ipynb# Main development notebook
└── README.md                  # Project documentation
