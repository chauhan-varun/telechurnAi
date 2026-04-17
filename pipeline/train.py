import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(input_path, model_path):
    df = pd.read_csv(input_path)
    X = df.drop('churned', axis=1)
    y = df['churned']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    model = RandomForestClassifier(n_estimators=350, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model
