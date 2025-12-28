import joblib
from src.preprocess import preprocess_data
import xgboost as xgb

DATA_PATH='data/raw/Dataset-1.csv'
MODEL_PATH='models/fraud_xgb.pkl'
SCALER_PATH='models/scaler.pkl'

def train():
    X_train, X_test, Y_train, Y_test,scaler=preprocess_data(DATA_PATH)

    model=xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=200,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
)

    model.fit(X_train,Y_train)

    joblib.dump(model,MODEL_PATH)
    joblib.dump(scaler,SCALER_PATH)

    print("Model and scaler saved successfully.")

if __name__=='__main__':
    train()
