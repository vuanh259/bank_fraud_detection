# Bank Account Fraud Detection using SMOTE and Decision Threshold Tuning

## Live Demo Web App
--> Experience the operational sandbox directly here: https://vustaz-bank-fraud-detection.hf.space


## Project Overview
This repository contains an end-to-end Machine Learning pipeline engineered to detect fraudulent bank account applications. Operating under extreme data imbalance typical in financial institutions (**1.1% fraud rate**), the project goes beyond vanilla metrics to optimize production threshold borders and provide model explainability (**XAI**).

The ultimate goal is to simulate a bank's automated clearing house decision-making framework, shifting from a generic "black-box" approach to an interpretable and operationally viable fraud prevention sandbox.

---

## Core Methodology & Machine Learning Pipeline

### 1. Exploratory Data Analysis & Preprocessing
* Automated high-cardinality nominal variable encoding using **One-Hot Encoding** to handle categorical fields safely without ordinal bias.
* Feature isolation based on production constraints (retaining variables accessible via instant APIs or real-time user-agent headers).

### 2. Imbalanced Data Resampling (SMOTE)
* Applied **SMOTE (Synthetic Minority Over-sampling Technique)** exclusively to the training set (`sampling_strategy=0.1`) to combat massive class skewness while strictly preventing **Data Leakage** into evaluation sets.

### 3. Model Benchmarking & Architecture Selection
* Built a robust baseline using **Random Forest Classifier** to assess feature importance scores.
* Conducted advanced benchmarking using **XGBoost Classifier (Gradient Boosting)** to enhance non-linear boundaries. XGBoost significantly optimized the Precision-Recall curve area, minimizing costly false alarms (False Positives) while maximizing catch rates.

### 4. Precision-Recall Curve & Threshold Tuning
* Abandoned the default generic `0.5` classification cutoff.
* Mapped the **Precision-Recall Curve** on independent test slices to identify the mathematical inflection point at **`0.2400`**, optimizing the banking risk appetite of "fail-safe over-omission".

---

## Model Performance Benchmark (Test Set)

| Model Architecture | Resampling Strategy | Evaluation Metric (ROC-AUC) | Optimal Decision Cutoff |
| :--- | :--- | :---: | :---: |
| **Random Forest** | SMOTE (`0.1`) | `0.8206` | `0.5000` (Default) |
| **XGBoost (Selected)** | SMOTE (`0.1`) | **👉 Optimized Highest** | **`0.2400` (Tuned)** |

---

## Production Tech Stack & Deployment
* **Core Language:** Python 3.x
* **Modeling & Math:** `scikit-learn`, `xgboost`, `numpy`, `pandas`
* **Explainable AI & Visualization:** `matplotlib`, `seaborn`, Feature Importance extraction.
* **Production Interface:** Packed via `joblib` and deployed as a micro-frontend using **Streamlit** on cloud architecture (**Hugging Face Spaces Engine**).

---

## Local Installation & Reproduction Guide

To run the simulation engine locally on your machine, execute the following workflow in your terminal:

```bash
# Clone the repository
git clone [https://github.com/vuanh259/bank_fraud_detection.git](https://github.com/vuanh259/bank_fraud_detection.git)
cd bank_fraud_detection

# Install required mathematical and deployment libraries
pip install -r requirements.txt

# Launch the Streamlit banking sandbox locally
streamlit run app.py