import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def encode_features(input_path, output_path):
    df = pd.read_csv(input_path)
    
    if 'contract_type' in df.columns:
        contract_dummies = pd.get_dummies(df['contract_type'], prefix='contract', drop_first=False)
        df = pd.concat([df, contract_dummies], axis=1)
        df.drop('contract_type', axis=1, inplace=True)
    
    if 'payment_method' in df.columns:
        payment_dummies = pd.get_dummies(df['payment_method'], prefix='payment', drop_first=False)
        df = pd.concat([df, payment_dummies], axis=1)
        df.drop('payment_method', axis=1, inplace=True)
    
    if 'zip_code' in df.columns:
        le_zip = LabelEncoder()
        df['zip_code_encoded'] = le_zip.fit_transform(df['zip_code'].astype(int))
        df.drop('zip_code', axis=1, inplace=True)
    
    if 'signup_month' in df.columns:
        df['signup_month'] = df['signup_month'].astype(int)
        month_dummies = pd.get_dummies(df['signup_month'], prefix='month', drop_first=False)
        df = pd.concat([df, month_dummies], axis=1)
        df.drop('signup_month', axis=1, inplace=True)
    
    df.to_csv(output_path, index=False)
    return df


def main():
    encode_features('telecom_churn_train_cleaned.csv', 'telecom_churn_train_encoded.csv')


if __name__ == "__main__":
    main()
