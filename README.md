# Bank Customer Churn Prediction

An end-to-end machine learning application that predicts whether a bank customer is likely to churn.

The application uses explainable AI (SHAP) to show why the model made each prediction, helping stakeholders understand key risk factors behind customer churn.

The project is deployed as an interactive Streamlit web app, which allows users to input customer attributes and instantly view:
- Churn probability
- Risk Classification
- Feature-level explanations

Live Demo link: (https://bank-churn-prediction-gchqtu8rprdwprnsqfjm66.streamlit.app/)


## Project Overview
Customer churn is a critical problem for banks, as acquiring new customers is far more expensive than retaining existing ones.

This project uses machine learning to:
- Predict customer churn probability
- Classify customers into High Risk / Low Risk
- Explain predictions using SHAP (SHapley Additive exPlanations)
- Present results through a clean and user-friendly web interface


## Objective
The objective of this project is to build a machine learning model that predicts whether a customer will churn (Exited = 1) or remain with the bank (Exited = 0) based on demographic and account-related features, while ensuring the predictions can be clearly explained and used in business decision-making.


## Dataset Description
- Source: Kaggle (Bank Customer Churn dataset)
- Size: ~10,000 customer records
- Target variable: Exited
- Feature types:
    - Demographic (Age, Gender, Geography)
    - Financial (Balance, Credit Score, Salary)
    - Behavioral (Active Membership, Number of Products)


## Exploratory Data Analysis (EDA)
Key insights from EDA include:
- No critical missing values after initial inspection.
- Significant class imbalance between churned and non-churned customers.
- Strong relationships between churn and features such as:
    - Age
    - Number of products
    - Active membership status

EDA was used to guide feature preparation and model selection, which not only for visualization.


## Feature Engineering & Data Preparation
- Removed non-informative identifier features (RowNumber, CustomerId and Surname) that do not contribute meaningful predictive signal
- Encoded categorical variables into numerical representations
- Scaled numerical features for models sensitive to feature magnitude (e.g., Logistic Regression, KNN, SVM)
- Ensured consistent feature ordering between training and deployment
- Prepared a unified feature set used across all evaluated models
- Split data into training and test sets to evaluate generalization on unseen data


## Model Training & Evaluation
### Models Evaluated
Multiple models were trained and evaluated to ensure a fair comparison:
- **Logistic Regression**
    Baseline linear model for interpretability and comparison

- **K-Nearest Neighbors (KNN)**
    Distance-based model to evaluate performance on normalized features

- **Support Vector Machine (SVM)**
    Margin-based classifier with feature scaling

- **Random Forest**
    Ensemble tree-based model to capture non-linear relationships

- **XGBoost**
    Boosted decision trees optimized for tabular data

### Evaluation Metrics
Model were compared using:
- **ROC-AUC**
- **F1-score (Churn class)**
- **Accuracy**

Given the class imbalance in churn data, F1-score and ROC-AUC were prioritized over accuracy.


## Model Selection
XGBoost was selected as the final model due to:
- Strong performance across evaluation metrics (ROC-AUC and F1-score)
- Robust handling of non-linear feature interactions
- Good generalization on unseen data
- Compatibility with SHAP for explainability


## Class Imbalance Handling
- Applied class weighting during model training to address imbalance between churned and non-churned customers
- Tuned the decision threshold to better balance precision and recall for churn prediction


## Model Explainability (SHAP)
This project emphasizes explainable AI, answering the question:
"Why did the model make this prediction?"

To address this, SHAP (SHapley Additive exPlanations) was used to interpret
individual predictions and highlight the contribution of each feature.

### Explainability Features
- SHAP waterfall plot for individual predictions
- Top 3 influencing factors (text explanation):
    - 🔴 Features that increased churn risk
    - 🔵 Features that reduced churn risk
- Toggleable explanation section in the web app

This improves transparency, trust and usability for non-technical stakeholders.


## Web Application (Streamlit)
The Streamlit app allows users to:
- Adjust customer attributes using sliders, radio buttons and dropdowns
- View churn probability and risk classification
- Explore SHAP-based explanations interactively

Example Inputs:
- Credit Score
- Age
- Tenure (Years)
- Balance
- Estimated Salary
- Number of Products
- Active Member
- Has Credit Card
- Gender
- Geography


## Deployment
- Deployed as an interactive web app using Streamlit Cloud
- Uses a fixed Python version and pinned dependencies for reliable execution
- Loads the trained XGBoost model using **XGBoost native** load_model method to avoid pickle-based serialization issues
- Applies a configurable decision threshold to balance precision and recall


## Project Structure
bank-churn-prediction/
│
├── app.py                      # Streamlit application
├── requirements.txt            # Pinned dependencies
├── runtime.txt                 # Python runtime for Streamlit Cloud
├── .gitignore                  # Git ignore rules
│
├── .streamlit/
│   └── config.toml             # Streamlit UI configuration
│
├── model/
│   ├── xgb_churn_model.json    # Trained XGBoost model
│   └── config.json             # Decision threshold
│
├── notebooks/
│   └── eda.ipynb               # EDA & model development
│
└── data/
    └── Churn_Modelling.csv     # Dataset


## Tech Stack
**Language**
- Python 3.11

**Machine Learning**
- XGBoost
- scikit-learn
- SHAP

**Data Processing & Visualization**
- pandas
- numpy
- Matplotlib

**Web & Deployment**
- Streamlit
- Streamlit Cloud

**Version Control**
- Git & GitHub


## Key Takeaways
This project demonstrates:
- End-to-end machine learning workflow (EDA → Modeling → Deployment)
- Handling of real-world class imbalance
- Explainable AI for transparent decision-making
- Clean UI/UX for non-technical users
- Production-aware deployment practices


## Future Improvements
- Global SHAP feature importance analysis
- Batch prediction via CSV upload
- Customer retention recommendation system
- Model monitoring and drift detection
- UI/UX enhancements


## Author
**Lim De Wei**
Computer Science (Data Science & AI)
Github: https://github.com/LimDeWei
Linkedln: https://www.linkedin.com/in/lim-de-wei-22809923a/
