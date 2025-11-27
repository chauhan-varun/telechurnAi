import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')


def process_test_data():
    print("Processing test dataset...")
    
    from clean_data import clean_dataset
    from encode_data import encode_features
    from feature_engineering import engineer_features
    
    clean_dataset('telecom_churn_test.csv', 'telecom_churn_test_cleaned.csv')
    encode_features('telecom_churn_test_cleaned.csv', 'telecom_churn_test_encoded.csv')
    engineer_features('telecom_churn_test_encoded.csv', 'telecom_churn_test_processed.csv')
    
    print("Test data processing complete!\n")


def predict_on_test():
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Using tuned model from 'best_model.pkl'")
    except FileNotFoundError:
        print("Tuned model not found. Training baseline Random Forest...")
        from train_model import train_and_evaluate
        results = train_and_evaluate()
        model = results['rf_model']
    
    test = pd.read_csv("telecom_churn_test_processed.csv")
    
    if 'churned' in test.columns:
        test = test.drop('churned', axis=1)
    
    print(f"\nPredicting on {len(test)} test samples...")
    test_predictions = model.predict(test)
    
    churn_count = test_predictions.sum()
    churn_rate = (churn_count / len(test)) * 100
    
    print(f"Predictions complete!")
    print(f"Predicted churners: {churn_count} ({churn_rate:.2f}%)")
    print(f"Predicted non-churners: {len(test) - churn_count} ({100-churn_rate:.2f}%)")
    
    output = pd.DataFrame({
        "customer_id": range(len(test)),
        "churn_prediction": test_predictions
    })
    
    output.to_csv("final_predictions.csv", index=False)
    print("\n✅ Prediction file saved to 'final_predictions.csv'")
    
    return output


def main():
    process_test_data()
    predictions = predict_on_test()
    return predictions


if __name__ == "__main__":
    main()
