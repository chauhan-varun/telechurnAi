import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')


def clean_dataset(file_path, output_path):
    df = pd.read_csv(file_path)
    
    df_cleaned = df.drop_duplicates()
    
    if 'income' in df_cleaned.columns and df_cleaned['income'].isnull().sum() > 0:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
        df_cleaned[numeric_cols] = knn_imputer.fit_transform(df_cleaned[numeric_cols])
    
    if 'nps_score' in df_cleaned.columns and df_cleaned['nps_score'].isnull().sum() > 0:
        median_nps = df_cleaned['nps_score'].median()
        df_cleaned['nps_score'].fillna(median_nps, inplace=True)
    
    remaining_missing = df_cleaned.isnull().sum()
    if remaining_missing.sum() > 0:
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() > 0:
                if df_cleaned[col].dtype in ['int64', 'float64']:
                    df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                else:
                    df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
    
    df_cleaned.to_csv(output_path, index=False)
    
    return df_cleaned


def main():
    clean_dataset('telecom_churn_train.csv', 'telecom_churn_train_cleaned.csv')


if __name__ == "__main__":
    main()
