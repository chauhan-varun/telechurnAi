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
        zip_bins = pd.qcut(df['zip_code_encoded'], q=5, labels=False, duplicates='drop')
        df['zip_region'] = zip_bins
    
    if 'age' in df.columns:
        contract_cols = [col for col in df.columns if col.startswith('contract_')]
        if contract_cols:
            for contract_col in contract_cols:
                df[f'age_x_{contract_col}'] = df['age'] * df[contract_col]
    
    if 'monthly_charges' in df.columns and 'tenure_months' in df.columns:
        df['charges_x_tenure'] = df['monthly_charges'] * df['tenure_months']
        
        median_charges = df['monthly_charges'].median()
        df['high_charges_new'] = ((df['monthly_charges'] > median_charges) & (df['tenure_months'] < 6)).astype(int)
        df['high_charges_old'] = ((df['monthly_charges'] > median_charges) & (df['tenure_months'] > 24)).astype(int)
    
    if 'num_support_calls' in df.columns and 'num_complaints' in df.columns:
        df['support_x_complaints'] = df['num_support_calls'] * df['num_complaints']
    
    if 'last_interaction_days' in df.columns and 'nps_score' in df.columns:
        df['inactivity_x_nps'] = df['last_interaction_days'] * df['nps_score']
    
    if 'income' in df.columns and 'monthly_charges' in df.columns:
        df['charges_to_income_ratio'] = df['monthly_charges'] / (df['income'] + 1)
    
    if 'data_usage_gb' in df.columns and 'has_internet' in df.columns:
        df['data_usage_x_internet'] = df['data_usage_gb'] * df['has_internet']
    
    df.to_csv(output_path, index=False)
    
    return df


def main():
    engineer_features(
        'telecom_churn_train_encoded.csv', 
        'telecom_churn_train_features.csv'
    )


if __name__ == "__main__":
    main()
