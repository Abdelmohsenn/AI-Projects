# Hand Gesture Recognition Model — ML1

> **Project Type:** Supervised Machine Learning Classification  
> **Dataset:** HaGRID Hand Gesture Landmarks (MediaPipe 21 landmarks, 3D) - [HaGRID Dataset](https://www.kaggle.com/datasets/kapitanov/hagrid)

### Here is a video illustrating the results: [Project Demo Video](https://drive.google.com/file/d/1CEhAqZW-axCK3SgaYa5Gfzsr7EMKBYQQ/view?usp=sharing)

---

## 1) Project Overview

This project performs **hand gesture classification** using **MediaPipe hand landmarks** extracted from the **HaGRID dataset**.  
Each sample contains **21 landmarks** with **(x, y, z)** coordinates → **63 numerical features**, and a **gesture label** as the target.

The workflow includes:
- Loading and exploring the dataset
- Normalizing landmarks
- Training 5 ML models (Random Forest, KNN, XGBoost, SVM, Ensemble-Voting)
- Comparing models using Test, Train accuracy, Precision, Recall, F1-score, and Confusion Matrix
- concluding with the best performing model and insights is implemented in `HandGestureModel.ipynb` with charts and plots.


## 2) Dataset

- **Input:** `Dataset/hand_landmarks_data.csv`
- **Features:** 63 values → 21 landmarks × (x, y, z)
- **Target:** `label` (gesture class)

**Shape summary:**
- 63 feature columns
- 1 label column → total 64 columns

### Data Balance
The dataset is generally balanced, with minor under-sampling for:
- `fist`
- `mute`

---

## 3) Preprocessing & Normalization

To make the model robust across different hand sizes and positions, landmarks were normalized per sample:

### Normalization steps
1. **Translation normalization:** subtract wrist landmark (landmark 0) from all landmarks  
2. **Scale normalization:** divide by hand size (norm of landmark 12) to normalize scale

---

## 4) Models Trained

The following models were trained and evaluated:

1 - **Random Forest**
2 - **KNN**
3 - **XGBoost**
4 - **SVM (RBF)**
5 - **Voting Classifier** — Ensemble of RF + XGB + KNN + SVM using soft voting

Each model was evaluated using:
- Train Accuracy
- Test Accuracy
- Precision
- Recall 
- F1-score 
- Confusion Matrix

---

## 5) Model Comparison

#### Model Performance Summary (Test Set)

| Model               | Test Accuracy |  Precision |     Recall |   F1-score |
| ------------------- | ------------: | ---------: | ---------: | ---------: |
| Random Forest       |        97.59% |     97.55% |     97.55% |     97.54% |
| KNN (n=4, distance) |        97.37% |     97.35% |     97.35% |     97.35% |
| XGBoost             |        98.05% |     98.01% |     98.03% |     98.02% |
| SVM (RBF)           |        98.38% |     98.32% |     98.35% |     98.33% |
| Voting (Soft)       |    **98.42%** | **98.37%** | **98.38%** | **98.37%** |

---

## 6) Repository Structure

```

.
├── Dataset
│   ├── hand_landmarks_data.csv
├── HandGestureModel.ipynb
├── assets/
│   ├── RandomForest/
│   ├── KNN/
│   ├── XGBoost/
│   ├── SVM/
│   └── VotingClassifier/
└── README.md

````
---

## 7) How to Run

### A) Install Dependencies
Create and activate a virtual environment, then install:

```bash
pip install -r requirements.txt
````

### B) Launch Notebook

```bash
jupyter notebook
```
---
