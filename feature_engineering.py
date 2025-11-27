import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def engineer_features(input_path, output_path):
    df = pd.read_csv(input_path)
    
    service_cols = [
        'has_internet', 'has_online_security', 'has_online_backup',
        'has_device_protection', 'has_tech_support', 'has_streaming_tv',
        'has_streaming_movies', 'has_phone', 'has_multiple_lines'
    ]
    
    available_service_cols = [col for col in service_cols if col in df.columns]
    
    if available_service_cols:
        df['num_services'] = df[available_service_cols].sum(axis=1)
    else:
        df['num_services'] = 0
    
    if 'monthly_charges' in df.columns:
        df['charge_per_service'] = df['monthly_charges'] / (df['num_services'] + 1)
    
    if 'tenure_months' in df.columns and 'monthly_charges' in df.columns:
        df['customer_lifetime_value'] = df['tenure_months'] * df['monthly_charges']
    
    if 'tenure_months' in df.columns:
        df['is_new'] = (df['tenure_months'] < 6).astype(int)
        df['is_loyal'] = (df['tenure_months'] > 24).astype(int)
    
    if 'monthly_charges' in df.columns and 'tenure_months' in df.columns:
        median_charges = df['monthly_charges'].median()
        df['high_risk_pricing'] = (
            (df['monthly_charges'] > median_charges) & 
            (df['tenure_months'] < 6)
        ).astype(int)
    
    if 'last_interaction_days' in df.columns:
        df['inactive_flag'] = (df['last_interaction_days'] > 30).astype(int)
    
    support_cols = []
    if 'num_support_calls' in df.columns:
        support_cols.append('num_support_calls')
    if 'num_complaints' in df.columns:
        support_cols.append('num_complaints')
    
    if support_cols:
        df['support_risk'] = df[support_cols].sum(axis=1)
    
    df.to_csv(output_path, index=False)
    
    return df


def main():
    engineer_features(
        'telecom_churn_train_encoded.csv', 
        'telecom_churn_train_features.csv'
    )


if __name__ == "__main__":
    main()
