LBS-XAI: A Hybrid Intelligent Framework for Explainable Hypertension Risk Prediction using SHAP-based Explainable AI
Overview
This project implements a multi-stage Machine Learning pipeline to predict the risk of high blood pressure. By integrating daily lifestyle habits with clinical biometric data, the system provides a comprehensive risk assessment. The application includes a Flask-based web dashboard and utilizes Explainable AI (SHAP) to provide transparent, feature-level insights for each prediction.

Hybrid Model Architecture
The LBS (Lifestyle-Biometric-Score) model operates in three distinct stages:

Stage 1 (Lifestyle): K-Means Clustering to identify lifestyle archetypes based on sleep, physical activity, and diet.

Stage 2 (Biometrics): XGBoost Classifier to determine risk probability using clinical markers such as BMI, glucose, and cholesterol.

Stage 3 (Meta-Model): Logistic Regression that synthesizes the outputs of previous stages into a final risk category.

Technical Stack
Backend: Python, Flask

Machine Learning: Scikit-Learn, XGBoost, SHAP

Data Processing: Pandas, NumPy

Frontend: HTML5, CSS3, JavaScript, Chart.js

Visualization: Matplotlib
