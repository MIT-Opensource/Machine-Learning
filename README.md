# 🧠 Machine Learning Group Assignment – MIT Group 3

This repository contains two machine learning projects developed as part of the *Machine Learning (IT4210B)* course at Mekelle Institute of Technology. The projects demonstrate the application of supervised learning techniques — classification and regression — on real-world datasets.

## 👥 Group 3 Members

| Name                | ID              |
|---------------------|-----------------|
| Mulugeta Gidey      | MIT/UR/175/11   |
| Aron T/haymanot     | MIT/UR/251/12   |
| Fanaye Tsegay       | MIT/UR/243/12   |
| Mehari Goitom       | MIT/UR/277/12   |
| Mahlet Atsbeha      | MIT/UR/323/12   |
| Aximawit Leake      | MIT/UR/018/12   |
| Lwam Tesfay         | MIT/UR/147/11   |
| Melat Kfle          | MIT/UR/099/12   |
| Frehiwot G/mariam   | MIT/UR/044/12   |
| Kidane Aregawi      | MIT/UR/310/12   |

---

## 📁 Project Overview

### 1. 🚢 Titanic Survival Classification

**Goal:** Predict whether a passenger survived the Titanic disaster using features like age, sex, class, and fare.

- **Algorithm:** Random Forest Classifier
- **Preprocessing:**
  - Filled missing age and embarked values
  - Dropped non-informative features
  - Label encoding
- **Evaluation Metrics:**
  - Precision: 83% (Died), 81% (Survived)
  - Recall: 88% (Died), 74% (Survived)
  - F1-score: 85% (Died), 77% (Survived)

### 2. 🧱 Concrete Strength Regression

**Goal:** Predict the compressive strength (MPa) of concrete based on its ingredients and age.

- **Algorithm:** Random Forest Regressor
- **Preprocessing:**
  - Standardized features
  - Train-test split (80/20)
- **Evaluation Metrics:**
  - RMSE: 5.52 MPa
  - MAE: 3.76 MPa
  - R² Score: 0.88

---

## 📊 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter/VS Code

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone git@github.com:MIT-Opensource/Machine-Learning.git
   cd Machine-Learning
