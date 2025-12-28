💳 Credit Card Fraud Detection System
📌 Overview

This project is an end-to-end machine learning system designed to detect fraudulent credit card transactions in real time.
It addresses extreme class imbalance (0.17% fraud) and prioritizes recall to minimize financial losses caused by undetected fraud.

The system includes:

Model training & evaluation

Real-time prediction API

Interactive dashboard

Explainable AI (SHAP)

🚀 Features

Handles highly imbalanced data using SMOTE

Trained with XGBoost for high recall and ROC-AUC

FastAPI backend for real-time inference

Streamlit dashboard for interactive predictions

SHAP explainability for model transparency

Clean separation of experimentation and production code

🧠 Tech Stack

Language: Python

Data Processing: Pandas, NumPy

Machine Learning: Scikit-learn, XGBoost, Imbalanced-learn

Explainability: SHAP

Backend API: FastAPI

Frontend: Streamlit

Version Control: Git & GitHub

📊 Dataset

Source: Kaggle – Credit Card Fraud Detection Dataset

Total Records: 284,807

Fraud Transactions: 492 (0.17%)

Features:

V1–V28: PCA-transformed features (confidentiality preserved)

Time: Seconds elapsed between transactions

Amount: Transaction amount

Class: 0 = Legitimate, 1 = Fraud

Due to confidentiality constraints, all sensitive features are PCA-transformed, which is common in financial datasets.

Folder Structure-
## 🏗️ Project Structure

- [`api/`](api/) – FastAPI backend
- [`app/`](app/) – Streamlit dashboard
- [`src/`](src/) – ML pipeline
- [`notebooks/`](notebooks/) – EDA & experiments
- [`models/`](models/) – Model artifacts (gitignored)
- [`data/`](data/) – Dataset files (gitignored)



📈 Model Performance

Recall (Fraud): ~95%

ROC-AUC: ~0.98

Focus Metric: Recall (to reduce false negatives)

Accuracy is misleading in highly imbalanced datasets; recall is prioritized to catch fraudulent transactions effectively.

🔍 Explainable AI (XAI)

The system integrates SHAP (SHapley Additive exPlanations) to interpret model predictions.

Explainability includes:

Global feature importance (overall fraud drivers)

Local explanations for individual transactions

Interactive explanations available in the Streamlit dashboard

This improves trust, transparency, and auditability in financial decision-making systems.

▶️ How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Train the Model
python -m src.train

3️⃣ Evaluate the Model
python -m src.evaluate

4️⃣ Run Explainability (SHAP)
python -m src.explain

5️⃣ Start FastAPI Backend
uvicorn api.main:app --reload


Open Swagger UI:

http://127.0.0.1:8000/docs

6️⃣ Launch Streamlit Dashboard
streamlit run app/streamlit_app.py

🌐 Live Demo

A public Streamlit application is deployed for real-time fraud prediction and explainability.
(Link added once active)
📌 Key Learnings

Handling extreme class imbalance is critical in fraud detection

Recall is more important than accuracy in high-risk domains

Explainability is essential for trust in ML systems

End-to-end deployment adds real-world value to ML projects

🧾 Resume Summary

Built an end-to-end credit card fraud detection system using XGBoost and SMOTE, achieving ~95% recall. Deployed real-time predictions with FastAPI, built an interactive Streamlit dashboard, and integrated SHAP-based explainability for transparent decision-making.

👤 Author

Giriprasath K
B.E. Computer Science Engineering (AI & ML)
