import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')


def process_test_data():
    from clean_data import clean_dataset
    from encode_data import encode_features
    from feature_engineering import engineer_features
    
    clean_dataset('telecom_churn_test.csv', 'telecom_churn_test_cleaned.csv')
    encode_features('telecom_churn_test_cleaned.csv', 'telecom_churn_test_encoded.csv')
    engineer_features('telecom_churn_test_encoded.csv', 'telecom_churn_test_processed.csv')


def predict_on_test():
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        from train_model import train_and_evaluate
        results = train_and_evaluate()
        model = results['rf_model']
    
    test = pd.read_csv("telecom_churn_test_processed.csv")
    
    if 'churned' in test.columns:
        test = test.drop('churned', axis=1)
    
    test_predictions = model.predict(test)
    
    output = pd.DataFrame({
        "customer_id": range(len(test)),
        "churn_prediction": test_predictions
    })
    
    output.to_csv("final_predictions.csv", index=False)
    
    return output


def main():
    process_test_data()
    predictions = predict_on_test()
    return predictions


if __name__ == "__main__":
    main()
