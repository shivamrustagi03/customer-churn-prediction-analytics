# 📉 Customer Churn Prediction & Retention Analytics

> A production-style end-to-end Machine Learning system for predicting telecom customer churn and identifying actionable customer retention strategies.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-End--to--End-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble%20Learning-success)
![Status](https://img.shields.io/badge/Project-Production--Style-success)

---

# 🚀 Project Overview

Customer churn is one of the biggest business challenges in subscription-based industries like telecom, banking, SaaS, and insurance.

Acquiring a new customer is significantly more expensive than retaining an existing one.

This project builds a **production-style Machine Learning pipeline** to:

* Predict whether a telecom customer is likely to churn
* Identify high-risk customers before they leave
* Understand the key churn drivers
* Generate business-focused retention recommendations
* Deploy predictions through an interactive Streamlit application

This project goes beyond prediction and focuses on **Customer Retention Analytics**, helping businesses make data-driven decisions.

---

# 🎯 Business Problem

Telecom companies lose revenue when customers discontinue their services.

The challenge is:

> **Can we proactively identify customers likely to churn and intervene before losing them?**

The objective of this project is to build an intelligent churn prediction system capable of identifying at-risk customers while providing interpretable insights into **why customers churn**.

---

# 🏗️ Architecture Diagram

```text
Raw Telecom Customer Data
            │
            ▼
     Data Cleaning
            │
            ▼
   Exploratory Data Analysis
            │
            ▼
    Feature Engineering
            │
            ▼
  Train-Test Split
            │
            ▼
 Handle Class Imbalance
       (SMOTEENN)
            │
            ▼
      Model Training
(Logistic Regression / Decision Tree /
 Random Forest / XGBoost)
            │
            ▼
 Hyperparameter Tuning
            │
            ▼
     Model Evaluation
(ROC-AUC, Precision, Recall,
 F1, Confusion Matrix)
            │
            ▼
     Model Explainability
      (Feature Importance
            + SHAP)
            │
            ▼
 Business Retention Insights
            │
            ▼
   Streamlit Deployment
```

---

# 🧠 Problem Statement

Build an end-to-end machine learning system that predicts whether a customer is likely to churn based on customer demographics, service usage, account information, and subscription behavior.

The final solution should:

✅ Predict churn probability

✅ Classify customer risk level

✅ Explain why churn occurs

✅ Help businesses take preventive action

---

# 📂 Dataset

### Dataset Used

**Telco Customer Churn Dataset**

The dataset contains telecom customer information such as:

* Customer demographics
* Internet services
* Payment methods
* Contract type
* Monthly charges
* Tenure
* Service subscriptions
* Churn status

### Target Variable

**Churn**

* `Yes` → Customer left service
* `No` → Customer retained

This is treated as a **Binary Classification Problem**.

---

# ⚙️ Tech Stack

## Languages & Libraries

### Core

* Python
* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-Learn
* XGBoost
* Imbalanced-Learn (SMOTEENN)
* SHAP

### Deployment

* Streamlit

---

# 🔍 Exploratory Data Analysis (EDA)

EDA was performed to understand:

* Customer churn patterns
* Missing values
* Class imbalance
* Service adoption trends
* Revenue-related behavior
* Churn distribution across demographics

### Ke
