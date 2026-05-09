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

### Key Business Insights

#### 1. Customers with month-to-month contracts churn more frequently

Long-term contracts significantly reduce churn risk.

#### 2. Customers with higher monthly charges are more likely to churn

Pricing sensitivity appears to influence retention.

#### 3. New customers show higher churn probability

Customer onboarding and early engagement are critical.

#### 4. Customers without additional services churn more often

Cross-selling services may improve retention.

---

# 🛠️ Data Preprocessing & Feature Engineering

The following preprocessing pipeline was implemented:

### Data Cleaning

* Removed inconsistencies
* Handled missing values
* Converted incorrect data types

### Feature Encoding

Categorical variables were converted into machine-readable format.

### Feature Engineering

Created transformed features for improved predictive performance.

### Train-Test Split

Dataset split into training and testing sets.

### Handling Class Imbalance

Customer churn datasets are typically imbalanced.

Instead of relying on raw class distribution, this project uses:

## SMOTEENN

SMOTEENN combines:

### SMOTE

Creates synthetic minority samples.

### ENN (Edited Nearest Neighbors)

Removes noisy examples.

This helps improve model generalization and reduces prediction bias.

**Why not accuracy only?**

Because churn prediction is an imbalanced classification problem.

Metrics like:

* ROC-AUC
* Recall
* Precision
* F1-score

provide a more realistic understanding of model performance.

---

# 🤖 Machine Learning Models

Multiple models were trained and compared.

## 1. Logistic Regression

Used as an interpretable baseline model.

### Why?

* Easy to interpret
* Strong baseline performance
* Helps understand linear relationships

---

## 2. Decision Tree

Used to capture non-linear customer behavior patterns.

### Why?

* Easy interpretability
* Captures feature interactions
* Simple rule-based learning

---

## 3. Random Forest

An ensemble learning model built using multiple decision trees.

### Why?

* Reduces overfitting
* Better generalization
* Handles non-linearity well
* Strong performance on tabular datasets

---

## 4. XGBoost

A gradient boosting ensemble model.

### Why?

* Strong predictive performance
* Effective on structured/tabular data
* Captures complex interactions

---

# 🔧 Hyperparameter Tuning

To improve performance:

### RandomizedSearchCV was used for:

* Random Forest
* XGBoost

This improves model performance while avoiding manual trial-and-error tuning.

---

# 📊 Model Performance

## Final Selected Model

# 🏆 Random Forest (Tuned)

Chosen based on:

* Highest business reliability
* Strong ROC-AUC performance
* Better balance between precision and recall
* Stable performance across churn classes

| Metric    | Score     |
| --------- | --------- |
| ROC-AUC   | **0.827** |
| Accuracy  | **0.759** |
| Precision | **0.535** |
| Recall    | **0.717** |
| F1 Score  | **0.613** |

### Why ROC-AUC matters?

Accuracy alone can be misleading in imbalanced datasets.

ROC-AUC measures how effectively the model separates churn vs non-churn customers.

A higher ROC-AUC indicates better ranking ability.

---

# 🧩 Model Explainability

Understanding **why** predictions happen is critical.

This project uses:

## Feature Importance

To identify the most influential churn drivers.

## SHAP (SHapley Additive Explanations)

SHAP helps explain:

* Why an individual customer is predicted to churn
* Which features influence predictions the most
* Positive/negative feature contributions

This improves:

✅ Model transparency

✅ Business trust

✅ Stakeholder communication

---

# 💡 Business Recommendations

Instead of stopping at prediction, this project focuses on retention strategies.

### High-Risk Customers

Action:

* Targeted retention campaigns
* Personalized offers
* Customer support outreach

---

### Month-to-Month Customers

Action:

* Encourage yearly contracts
* Loyalty discounts

---

### High Monthly Charges

Action:

* Customized pricing plans
* Bundle discounts

---

### New Customers

Action:

* Better onboarding
* Engagement campaigns
* Customer education

---

# 🖥️ Streamlit Application

The project includes an interactive Streamlit web application.

### Features

✅ Customer churn prediction

✅ Churn probability score

✅ Risk classification

* Low Risk
* Medium Risk
* High Risk

✅ Model performance overview

✅ Feature importance visualization

---

## 📸 Application Screenshots

### Homepage

![Homepage](images/Customer%20Churn%20Homepage.png)

---

### Prediction Output

![Prediction](images/Customer%20Churn%20Prediciton%20Output.png)

---

# 📁 Project Structure

```text
customer-churn-prediction-analytics/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── best_model.pkl
│
├── streamlit_app/
│   └── app.py
│
├── reports/
│   └── screenshots/
│
├── requirements.txt
└── README.md
```

---

# ⚡ Installation & Usage

## Clone Repository

```bash
git clone https://github.com/shivamrustagi03/customer-churn-prediction-analytics.git
cd customer-churn-prediction-analytics
```

## Create Virtual Environment

```bash
python -m venv churn
```

### Activate Environment

Windows:

```bash
churn\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train Model

```bash
python -m src.train
```

## Run Streamlit App

```bash
streamlit run streamlit_app/app.py
```

---

# 🎯 Key Project Highlights

✔ End-to-end Machine Learning pipeline

✔ Professional EDA & Feature Engineering

✔ Class imbalance handling using SMOTEENN

✔ Multi-model comparison

✔ Hyperparameter tuning

✔ SHAP Explainability

✔ Business-oriented retention analytics

✔ Streamlit deployment

✔ Modular project structure

✔ Production-style workflow

---

# 💼 Skills Demonstrated

### Data Science

* Exploratory Data Analysis (EDA)
* Feature Engineering
* Data Preprocessing
* Machine Learning
* Model Evaluation
* Explainable AI

### Machine Learning

* Classification Models
* Ensemble Learning
* Hyperparameter Tuning
* Imbalanced Data Handling

### Business Analytics

* Customer Retention Analytics
* Churn Analysis
* Business Recommendations

### Deployment

* Streamlit App Development

---

# 🧪 Future Improvements

Potential enhancements:

* Model monitoring
* MLflow experiment tracking
* Real-time prediction API
* Cloud deployment
* Advanced feature engineering
* Customer segmentation

---

# 👨‍💻 Author

**Shivam Rustagi**

Aspiring Data Scientist passionate about Machine Learning, Analytics, and solving real-world business problems through data.

### GitHub

[https://github.com/shivamrustagi03](https://github.com/shivamrustagi03)

---

## ⭐ If you found this project useful, consider starring the repository.
