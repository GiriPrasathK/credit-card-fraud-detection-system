import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from src.preprocess import preprocess_data

DATA_PATH='data/raw/Dataset-1.csv'
MODEL_PATH='models/fraud_xgb.pkl'

X_train, X_test, Y_train, Y_test,scaler=preprocess_data(DATA_PATH)
model=joblib.load(MODEL_PATH)

y_pred=model.predict(X_test)
y_prob=model.predict_proba(X_test)[:,1]

print("Classification Report:")
print(classification_report(Y_test,y_pred))
print("ROC-AUC Score:",roc_auc_score(Y_test,y_prob))

