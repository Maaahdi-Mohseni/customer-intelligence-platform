# Customer Intelligence Platform

An end-to-end Machine Learning project focused on predicting customer churn in the telecommunications industry.

## Overview

Customer churn is a major challenge for subscription-based businesses. Understanding which customers are likely to leave a service can help businesses make better decisions and develop effective customer retention strategies.

This project aims to build a machine learning platform that analyzes customer data and predicts the likelihood of customer churn.

The project is being developed with a focus on **Machine Learning Engineering**, including modular code architecture, model development, API integration, database management, and deployment.

## Dataset

The project uses the [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

The dataset contains customer information such as:

- Customer demographics
- Account and contract information
- Internet and phone services
- Payment methods
- Monthly and total charges
- Customer churn status

## Current Progress

- [x] Exploratory Data Analysis (EDA)
- [x] Data preprocessing and feature transformation
- [x] Train-test split and cross-validation
- [x] Model comparison
- [x] Logistic Regression model development
- [x] Hyperparameter tuning with GridSearchCV
- [x] Modular project structure
- [ ] Model evaluation and finalization in the modular pipeline
- [ ] FastAPI backend development
- [ ] PostgreSQL database integration
- [ ] Streamlit frontend development
- [ ] Testing
- [ ] Dockerization and deployment

## Machine Learning

The initial model development phase included comparing multiple algorithms:

- Logistic Regression
- Decision Tree
- Random Forest

Logistic Regression was selected based on its performance during cross-validation.

Model evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Hyperparameter optimization is performed using GridSearchCV with Stratified K-Fold Cross-Validation.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- FastAPI (planned)
- PostgreSQL (planned)
- Streamlit (planned)
- Pytest (planned)
- Docker (planned)

## Project Structure

```text
customer-intelligence-platform/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│   ├── data/
│   │   └── load_data.py
│   │
│   ├── features/
│   │   └── preprocess.py
│   │
│   └── models/
│       ├── train.py
│       └── tune.py
│
├── tests/
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Project Goal

The ultimate goal is to develop a complete, modular, and maintainable machine learning application that can process customer data, generate churn predictions, and expose those predictions through an interactive application.

---

**Status:** Under Development
