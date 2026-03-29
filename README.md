# 🎧 Spotify Popularity Prediction

📄 **Full Report:** [CS171.pdf](./CS171.pdf)

---

## 📌 Overview

This project explores how machine learning can be used to predict whether a song will become popular on Spotify. Using a real-world dataset, we build and evaluate multiple models with a focus on handling **imbalanced data** and optimizing for meaningful metrics like **PR-AUC and recall**.

The goal is not just accuracy, but correctly identifying **rare popular songs**, which better reflects real-world recommendation systems.

---

## ❓ Problem Statement

- Predict whether a song is **popular (top 20%) or not**
- Handle **imbalanced data** (~78% non-popular, ~22% popular)
- Optimize for:
  - **Recall** (catch popular songs)
  - **PR-AUC** (performance on imbalanced data)

---

## 📊 Dataset & Features

The dataset includes Spotify track, artist, and album information.

### Features Used:
- Track number  
- Track duration (log transformed)  
- Explicit (binary encoded)  
- Artist popularity  
- Artist followers (log transformed)  
- Album total tracks  
- Album release year  
- Album type  
- Artist genres  

### Key Notes:
- Popularity converted into **binary classification (top 20%)**
- Avoided **data leakage** (excluded raw popularity score)
- Preserved real-world imbalance instead of artificially balancing

---

## ⚙️ Methodology

### Data Preprocessing
- Log transformations for skewed features  
- Feature encoding (binary + categorical)  
- Train/Validation/Test split: **80 / 10 / 10**  
- Class weighting to handle imbalance  

### PCA (Principal Component Analysis)
- Tested but **not used**
- Reduced performance by removing important signals for the minority class

---

## 🤖 Models

### 1. Neural Network (MLP - Main Model)

**Baseline Model:**
- 2 hidden layers, 10 units each  
- ReLU activation  
- SGD optimizer  
- Binary Cross Entropy  

**Best Model (after tuning):**
- 4 hidden layers  
- 20 units per layer  
- ReLU activation  
- Glorot-normal initializer  
- SGD optimizer (learning rate = 0.005)  
- No dropout  

---

### 2. Other Models Tested

- Support Vector Machine (SVM)  
- Random Forest  

---

## 📈 Results

### Final Neural Network (MLP)

- Accuracy: **0.5478**  
- Precision: **0.3109**  
- Recall: **0.9048**  
- F1 Score: **0.4628**  
- ROC-AUC: **0.7252**  
- PR-AUC: **0.3474**  

➡️ High recall ensures most popular songs are detected, even with lower accuracy.

---

### SVM

- Accuracy: **0.6310**  
- Recall: **0.8783**  
- PR-AUC: **0.4574**  

---

### Random Forest

- Accuracy: **0.7893**  
- Recall: **0.2751**  
- PR-AUC: **0.4683**  

---

## 🧠 Key Insights

- **Imbalanced data is critical** — do not artificially balance it  
- **PR-AUC is more important than accuracy** for this problem  
- Neural networks struggled with generalization compared to simpler models  
- **SVM achieved the best balance of performance metrics**  
- Random Forest had high accuracy but missed many popular songs  

---

## ⚠️ Tradeoffs

- High recall leads to more false positives  
- Lower accuracy is acceptable to avoid missing popular songs  
- Reflects real-world recommendation systems (e.g., Spotify)

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- Scikit-learn  
- Pandas / NumPy  
- Matplotlib  

---

## 🚀 How to Run

```bash
git clone https://github.com/jonehthan/spotify-popularity-report.git
cd spotify-popularity-report
pip install -r requirements.txt
jupyter notebook
