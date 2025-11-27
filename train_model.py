import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def train_and_evaluate():
    df = pd.read_csv('telecom_churn_train_features.csv')
    
    X = df.drop('churned', axis=1)
    y = df['churned']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    log_model = LogisticRegression(max_iter=3000, random_state=42)
    log_model.fit(X_train, y_train)
    log_preds = log_model.predict(X_val)
    rf_model = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_val)
    
    log_f1 = f1_score(y_val, log_preds, average='weighted')
    rf_f1 = f1_score(y_val, rf_preds, average='weighted')
    
    return {
        'log_model': log_model,
        'rf_model': rf_model,
        'log_f1': log_f1,
        'rf_f1': rf_f1,
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val
    }


def main():
    results = train_and_evaluate()
    return results


if __name__ == "__main__":
    main()
