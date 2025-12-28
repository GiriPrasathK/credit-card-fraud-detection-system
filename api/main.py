import joblib
from pydantic import BaseModel
from fastapi import FastAPI
import os
import numpy as np
from typing import List

app=FastAPI(title="Credit Card Fraud Detection API")
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
MODEL_PATH=os.path.join(BASE_DIR,'..','models','fraud_xgb.pkl')     
SCALER_PATH=os.path.join(BASE_DIR,'..','models','scaler.pkl')

model=joblib.load(MODEL_PATH)
scaler=joblib.load(SCALER_PATH)

class Transaction(BaseModel):
    values:List[float]

@app.get("/health")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict(transaction: Transaction):
    data=np.array([list(transaction.dict().values())]).reshape(1,-1)
    data_scaled=scaler.transform(data)

    fraud_prob=model.predict_proba(data_scaled)[:,1][0]
    fraud_label=int(fraud_prob>=0.5)

    return{
        "fraud_probability": round(float(fraud_prob), 4),
        "fraud_prediction": fraud_label
    }  
    
    