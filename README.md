# TeleChurnAi - Customer Churn Prediction

Machine learning pipeline for predicting customer churn in telecom industry.

## 📊 Project Overview

- **Dataset**: 6,048 training samples, 2,045 test samples
- **Features**: 64 (after engineering)
- **Best Model**: Random Forest with 92% accuracy
- **Target**: Predict customer churn (binary classification)

## 🚀 Quick Start (VM with 24GB RAM)

### 1. Setup Environment

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### 2. Complete Pipeline (Training + Prediction)

```bash
# Run entire pipeline: clean → encode → engineer → train → predict
uv run python main.py
```

This will:
- Clean and process training data
- Encode categorical features
- Engineer 20 new features
- Train Logistic Regression and Random Forest models
- Process test data
- Generate `final_predictions.csv`

### 3. Individual Steps

#### Step 1: Data Cleaning
```bash
uv run python clean_data.py
```
- Removes duplicates
- Handles missing values (KNN imputation for income, median for NPS)
- Output: `telecom_churn_train_cleaned.csv`

#### Step 2: Feature Encoding
```bash
uv run python encode_data.py
```
- One-hot encoding for contract_type, payment_method, signup_month
- Label encoding for zip_code
- Output: `telecom_churn_train_encoded.csv`

#### Step 3: Feature Engineering
```bash
uv run python feature_engineering.py
```
- Creates 20 derived features (service bundles, usage intensity, LTV, etc.)
- Output: `telecom_churn_train_features.csv`

#### Step 4: Model Training (Baseline)
```bash
uv run python train_model.py
```
- Trains Logistic Regression and Random Forest
- 80/20 train-validation split
- Expected F1 Score: ~0.91

#### Step 5: Hyperparameter Tuning (Optional - Takes ~30-40 minutes)
```bash
uv run python tune_model.py
```
- GridSearchCV with 3-fold cross-validation
- Tests 108 parameter combinations
- Saves best model to `best_model.pkl`
- **Recommended for VM with 24GB RAM**

#### Step 6: Test Predictions
```bash
uv run python predict_test.py
```
- Processes test data through entire pipeline
- Uses best_model.pkl (if available) or baseline Random Forest
- Output: `final_predictions.csv`

## 📁 Project Structure

```
TeleChurnAi/
├── telecom_churn_train.csv          # Raw training data
├── telecom_churn_test.csv           # Raw test data
├── clean_data.py                    # Data cleaning
├── encode_data.py                   # Feature encoding
├── feature_engineering.py           # Feature engineering
├── train_model.py                   # Baseline model training
├── tune_model.py                    # Hyperparameter tuning
├── predict_test.py                  # Test predictions
├── main.py                          # Complete pipeline
└── final_predictions.csv            # Output predictions
```

## 🎯 Expected Results

### Baseline Random Forest
- **Accuracy**: 92%
- **F1 Score**: 0.914
- **Precision (Churn)**: 96%
- **Recall (Churn)**: 71%

### After Hyperparameter Tuning
- **F1 Score**: ~0.91-0.92 (similar or slightly better)
- **Best Parameters**: Typically `max_depth=None, n_estimators=300-500`

## 🔧 Engineered Features (20 Total)

1. **Service Bundles**: num_services, has_streaming
2. **Charge Ratios**: charge_per_service, charges_to_income_ratio
3. **Usage Intensity**: data_per_month, calls_per_month, sms_per_month
4. **Tenure Segments**: is_new_customer, is_long_term
5. **Lifetime Value**: approx_ltv
6. **Geographical**: zip_region
7. **Interactions**: age_x_contract (3), charges_x_tenure, high_charges_new/old, support_x_complaints, inactivity_x_nps, data_usage_x_internet

## 💾 Output Files

- `telecom_churn_train_cleaned.csv` - Cleaned training data
- `telecom_churn_train_encoded.csv` - Encoded training data
- `telecom_churn_train_features.csv` - Training data with engineered features
- `telecom_churn_test_processed.csv` - Processed test data
- `best_model.pkl` - Tuned Random Forest model (if tuning was run)
- `final_predictions.csv` - **Final predictions for submission**

## ⚡ Performance Tips for VM

1. **Use all CPU cores**: Models use `n_jobs=-1` by default
2. **Hyperparameter tuning**: Run `tune_model.py` for best results (takes 30-40 min)
3. **Memory**: 24GB RAM is sufficient for all operations
4. **Skip tuning**: If time-constrained, baseline Random Forest achieves 92% accuracy

## 📝 Final Predictions Format

```csv
customer_id,churn_prediction
0,0
1,1
2,0
...
```

- `0` = No churn
- `1` = Churn

## 🐛 Troubleshooting

**Issue**: Out of memory during tuning
- **Solution**: Reduce parameter grid in `tune_model.py`

**Issue**: Missing dependencies
- **Solution**: Run `uv sync` to install all packages

**Issue**: Feature mismatch error
- **Solution**: Ensure test data goes through same pipeline (clean → encode → engineer)

## 📊 Model Evaluation Metrics

The models are evaluated using:
- **F1 Score (weighted)**: Primary metric (handles class imbalance)
- **Precision**: How many predicted churners actually churned
- **Recall**: How many actual churners were caught
- **Confusion Matrix**: Detailed breakdown of predictions

## 🎓 Next Steps

1. Run complete pipeline on VM: `uv run python main.py`
2. (Optional) Run hyperparameter tuning: `uv run python tune_model.py`
3. Submit `final_predictions.csv`
