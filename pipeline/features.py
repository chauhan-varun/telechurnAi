import pandas as pd
import numpy as np

def engineer_features(input_path, output_path):
    df = pd.read_csv(input_path)
    service_cols = ['has_internet', 'has_online_security', 'has_online_backup', 'has_device_protection', 'has_tech_support', 'has_streaming_tv', 'has_streaming_movies', 'has_phone', 'has_multiple_lines']
    available_service_cols = [col for col in service_cols if col in df.columns]
    
    df['num_services'] = df[available_service_cols].sum(axis=1) if available_service_cols else 0
    if 'has_streaming_tv' in df.columns and 'has_streaming_movies' in df.columns:
        df['has_streaming'] = ((df['has_streaming_tv'] == 1) | (df['has_streaming_movies'] == 1)).astype(int)
    if 'monthly_charges' in df.columns:
        df['charge_per_service'] = df['monthly_charges'] / (df['num_services'] + 0.1)
    if 'data_usage_gb' in df.columns and 'tenure_months' in df.columns:
        df['data_per_month'] = df['data_usage_gb'] / (df['tenure_months'] + 0.1)
    if 'minutes_used' in df.columns and 'tenure_months' in df.columns:
        df['calls_per_month'] = df['minutes_used'] / (df['tenure_months'] + 0.1)
    if 'sms_sent' in df.columns and 'tenure_months' in df.columns:
        df['sms_per_month'] = df['sms_sent'] / (df['tenure_months'] + 0.1)
    if 'tenure_months' in df.columns:
        df['is_new_customer'] = (df['tenure_months'] < 3).astype(int)
        df['is_long_term'] = (df['tenure_months'] > 24).astype(int)
    if 'tenure_months' in df.columns and 'monthly_charges' in df.columns:
        df['approx_ltv'] = df['tenure_months'] * df['monthly_charges']
    if 'zip_code_encoded' in df.columns:
        df['zip_region'] = pd.qcut(df['zip_code_encoded'], q=5, labels=False, duplicates='drop')
    
    df.to_csv(output_path, index=False)
    return df
