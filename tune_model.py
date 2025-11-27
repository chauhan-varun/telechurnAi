import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')


def tune_hyperparameters():
    df = pd.read_csv('telecom_churn_train_features.csv')
    
    from sklearn.model_selection import train_test_split
    X = df.drop('churned', axis=1)
    y = df['churned']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    params = {
        "n_estimators": [300, 500, 700],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }
    
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        params,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=2
    )
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    
    preds = best_model.predict(X_val)
    val_f1 = f1_score(y_val, preds, average='weighted')
    
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    return best_model, grid.best_params_, val_f1


def main():
    best_model, best_params, f1 = tune_hyperparameters()
    return best_model, best_params, f1


if __name__ == "__main__":
    main()
