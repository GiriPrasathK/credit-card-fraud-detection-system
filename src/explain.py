import joblib
import shap
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fraud_xgb.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'Dataset-1.csv')

model=joblib.load(MODEL_PATH)
scaler=joblib.load(SCALER_PATH)

df=pd.read_csv(DATA_PATH)   
x=df.drop('Class',axis=1)

N=500

x_scaled=scaler.transform(x)
x_scaled=x_scaled[:N]
#------SHAp EXPLAINER------#
explainer=shap.Explainer(model)
shap_values=explainer(x_scaled)

shap.summary_plot(shap_values, x_scaled[:1000], feature_names=x.columns)

