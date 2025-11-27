# TeleChurnAi - Customer Churn Prediction 📊

An AI-powered customer churn prediction system for telecom companies, achieving **98.1% accuracy** in identifying customers at risk of churning. Built with machine learning and an interactive Streamlit dashboard for real-time insights and predictions.

## 🎯 Overview

TeleChurnAi helps telecom companies predict and prevent customer churn by analyzing customer behavior, usage patterns, and service interactions. The system processes raw customer data through a complete ML pipeline and provides actionable insights through an intuitive web interface.

### Key Features

- **High Accuracy Predictions**: 98.1% accuracy using Random Forest classifier
- **Complete ML Pipeline**: Automated data cleaning, encoding, feature engineering, and model training
- **Interactive Dashboard**: Real-time visualizations and insights using Streamlit
- **Batch Predictions**: Upload CSV files for bulk churn predictions
- **Risk Analysis**: Identify high-risk customers with probability scores
- **Feature Insights**: Understand key drivers of customer churn
- **Data Pipeline Visualization**: See how raw data transforms into ML-ready features

## 🚀 Tech Stack

- **Python 3.12+**
- **Machine Learning**: scikit-learn (Random Forest, Logistic Regression)
- **Data Processing**: pandas, numpy
- **Visualization**: Streamlit, Plotly
- **Package Management**: uv

## 📦 Installation

### Prerequisites

- Python 3.12 or higher
- uv package manager (recommended) or pip

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/chauhan-varun/telechurnAi.git
   cd telechurnAi
   ```

2. **Install dependencies using uv** (recommended)
   ```bash
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python --version  # Should be 3.12+
   ```

## 🎮 Usage

### Running the Complete ML Pipeline

To process data, train the model, and generate predictions:

```bash
python main.py
```

This will execute the entire pipeline:
1. Clean the training data
2. Encode categorical features
3. Engineer new features
4. Train the model
5. Process test data and generate predictions

### Launching the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Individual Pipeline Steps

You can also run individual steps:

```bash
# Data cleaning
python clean_data.py

# Feature encoding
python encode_data.py

# Feature engineering
python feature_engineering.py

# Model training
python train_model.py

# Generate predictions on test data
python predict_test.py
```

## 📊 Dashboard Features

### 1. **Overview Tab**
- Churn distribution pie chart
- Risk score distribution histogram
- Confusion matrix visualization
- Key performance metrics (Accuracy, Precision, Recall, F1)

### 2. **Data Pipeline Tab**
- Step-by-step visualization of data transformation
- Shows raw → cleaned → encoded → engineered features
- Metrics at each pipeline stage

### 3. **Live Prediction Tab**
- Upload CSV files for batch predictions
- Get churn probability for each customer
- Download predictions as CSV

### 4. **High Risk Customers Tab**
- Top 20 customers most likely to churn
- Risk scores and key customer attributes
- Revenue at risk calculations

### 5. **Feature Insights Tab**
- Top 15 most important features for churn prediction
- Feature importance visualization
- Key insights and drivers

## 🔧 Project Structure

```
TeleChurnAi/
├── app.py                              # Streamlit dashboard
├── main.py                             # Complete pipeline orchestrator
├── clean_data.py                       # Data cleaning module
├── encode_data.py                      # Feature encoding module
├── feature_engineering.py              # Feature engineering module
├── train_model.py                      # Model training module
├── tune_model.py                       # Hyperparameter tuning
├── predict_test.py                     # Test data prediction
├── best_model.pkl                      # Trained model (generated)
├── telecom_churn_train.csv             # Training data (raw)
├── telecom_churn_test.csv              # Test data (raw)
├── telecom_churn_train_cleaned.csv     # Cleaned training data
├── telecom_churn_train_encoded.csv     # Encoded training data
├── telecom_churn_train_features.csv    # Final training features
├── telecom_churn_test_processed.csv    # Processed test data
├── final_predictions.csv               # Test predictions output
├── pyproject.toml                      # Project dependencies
└── README.md                           # This file
```

## 🤖 ML Pipeline Details

### 1. Data Cleaning (`clean_data.py`)
- Removes duplicate records (~3% of data)
- Handles missing values:
  - KNN imputation for `income`
  - Median imputation for `nps_score`
  - Standard imputation for other columns

### 2. Feature Encoding (`encode_data.py`)
- **One-Hot Encoding**: `contract_type`, `payment_method`, `signup_month`
- **Label Encoding**: `zip_code`

### 3. Feature Engineering (`feature_engineering.py`)
Creates 20+ derived features:
- **Service Bundles**: Total services, premium services
- **Charge Ratios**: Overage/monthly charges, avg monthly charges
- **Usage Intensity**: Data, call, SMS usage metrics
- **Tenure Segments**: Customer lifecycle stages
- **Interaction Features**: Cross-feature combinations

### 4. Model Training (`train_model.py`)
- **Algorithm**: Random Forest Classifier
- **Hyperparameters**: 350 trees, unlimited depth
- **Validation**: 80/20 train-validation split
- **Metrics**: Accuracy, Precision, Recall, F1 Score

### 5. Hyperparameter Tuning (`tune_model.py`)
- Grid search for optimal parameters
- Cross-validation for robust performance
- Saves best model as `best_model.pkl`

## 📈 Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 98.1%  |
| Precision | 99.7%  |
| Recall    | 92.6%  |
| F1 Score  | 0.960  |

## 💡 Use Cases

1. **Proactive Retention**: Identify at-risk customers before they churn
2. **Targeted Marketing**: Focus retention campaigns on high-risk segments
3. **Revenue Protection**: Calculate and protect revenue at risk
4. **Customer Insights**: Understand key factors driving churn
5. **Strategic Planning**: Make data-driven decisions on service improvements

## 🔮 Making Predictions

### Using the Dashboard
1. Navigate to the "Live Prediction" tab
2. Upload a CSV file with customer data
3. Click "Predict Churn"
4. Download results with risk scores

### Using Python
```python
import pandas as pd
import pickle

# Load model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load and process your data
# (must go through clean → encode → engineer pipeline)

# Make predictions
predictions = model.predict(processed_data)
probabilities = model.predict_proba(processed_data)[:, 1]
```

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy with one click

### VM Deployment
1. Install dependencies on VM
2. Run with nohup:
   ```bash
   nohup streamlit run app.py --server.port 8501 &
   ```
3. Configure firewall to allow port 8501

## 📝 Data Requirements

Your CSV file should include columns such as:
- Customer demographics (age, income, etc.)
- Service usage (data, calls, SMS)
- Account information (tenure, contract type, payment method)
- Support interactions (number of calls, complaints)
- Billing information (monthly charges, overage charges)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Built with ❤️ for telecom analytics and customer retention

---

**TeleChurn AI** - Predict. Prevent. Retain.
