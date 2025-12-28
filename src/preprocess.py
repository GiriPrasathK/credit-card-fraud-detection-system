import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os
import urllib.request

def preprocess_data(file_path):
    # Load the dataset
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        url="https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
        urllib.request.urlretrieve(url, file_path)


    df = pd.read_csv(file_path)
    
    # Separate features and target variable
    X = df.drop('Class', axis=1)
    Y = df['Class']
    
    # Split the dataset into training and testing sets with stratification
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    # Standardize the feature values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE to handle class imbalance in the training set
    smote = SMOTE(random_state=42)
    X_train_res, Y_train_res = smote.fit_resample(X_train_scaled, Y_train)
    
    return X_train_res, X_test_scaled, Y_train_res, Y_test,scaler
