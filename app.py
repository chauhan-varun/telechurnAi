import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile
import warnings
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

# Import modular pipeline functions
from pipeline.clean import clean_dataset
from pipeline.encode import encode_features
from pipeline.features import engineer_features
from pipeline.train import train_model

warnings.filterwarnings('ignore')

st.set_page_config(page_title="TeleChurn AI", layout="wide", page_icon="📊")

# --- DATA LOADING ---

@st.cache_data
def load_data():
    if os.path.exists('data/processed/telecom_churn_train_features.csv'):
        return pd.read_csv('data/processed/telecom_churn_train_features.csv')
    return None

@st.cache_resource
def load_model():
    if os.path.exists('models/best_model.pkl'):
        with open('models/best_model.pkl', 'rb') as f:
            return pickle.load(f)
    return None

def get_metrics(model, X, y):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds),
        'recall': recall_score(y, preds),
        'f1': f1_score(y, preds),
        'predictions': preds,
        'probabilities': proba
    }

def predict_churn_single(model, data):
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = os.path.join(tmpdir, 'raw.csv')
        cleaned_path = os.path.join(tmpdir, 'cleaned.csv')
        encoded_path = os.path.join(tmpdir, 'encoded.csv')
        final_path = os.path.join(tmpdir, 'final.csv')
        
        data.to_csv(raw_path, index=False)
        clean_dataset(raw_path, cleaned_path)
        encode_features(cleaned_path, encoded_path)
        engineer_features(encoded_path, final_path)
        
        processed_data = pd.read_csv(final_path)
        if 'churned' in processed_data.columns:
            processed_data = processed_data.drop('churned', axis=1)
        
        predictions = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)[:, 1]
        return predictions, probabilities

# --- UI LAYOUT ---

st.title("📊 TeleChurn AI - Customer Churn Prediction")
st.markdown("### Predict and prevent customer churn with 92% accuracy")
st.markdown("---")

train_data = load_data()
model = load_model()

if train_data is not None and model is not None:
    X = train_data.drop('churned', axis=1)
    y = train_data['churned']
    metrics = get_metrics(model, X, y)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("🎯 Accuracy", f"{metrics['accuracy']*100:.1f}%")
    with col2: st.metric("🎯 Precision", f"{metrics['precision']*100:.1f}%")
    with col3: st.metric("🎯 Recall", f"{metrics['recall']*100:.1f}%")
    with col4: st.metric("🎯 F1 Score", f"{metrics['f1']:.3f}")
    with col5: st.metric("👥 Total Customers", f"{len(train_data):,}")

    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔧 Data Pipeline", "🔮 Live Prediction", "⚠️ High Risk Customers", "📈 Feature Insights"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Churn Distribution")
            churn_counts = train_data['churned'].value_counts()
            fig = px.pie(values=churn_counts.values, names=['No Churn', 'Churn'], color_discrete_sequence=['#00CC96', '#EF553B'], hole=0.4)
            st.plotly_chart(fig, width="stretch")
            st.metric("Churn Rate", f"{(churn_counts[1]/len(train_data)*100):.1f}%")
        with col2:
            st.subheader("Risk Score Distribution")
            fig = px.histogram(x=metrics['probabilities'], nbins=50, labels={'x': 'Churn Probability', 'y': 'Count'}, color_discrete_sequence=['#636EFA'])
            fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
            st.plotly_chart(fig, width="stretch")
        
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, metrics['predictions'])
        fig = go.Figure(data=go.Heatmap(z=cm, x=['Predicted No Churn', 'Predicted Churn'], y=['Actual No Churn', 'Actual Churn'], colorscale='Blues', text=cm, texttemplate='%{text}', textfont={"size": 20}))
        st.plotly_chart(fig, width="stretch")

    with tab2:
        st.subheader("⚙️ Data Processing Pipeline")
        if st.button("🚀 Run Full Pipeline", type="primary"):
            with st.status("Processing data..."):
                st.write("Cleaning data...")
                clean_dataset('data/raw/telecom_churn_train.csv', 'data/processed/telecom_churn_train_cleaned.csv')
                st.write("Encoding features...")
                encode_features('data/processed/telecom_churn_train_cleaned.csv', 'data/processed/telecom_churn_train_encoded.csv')
                st.write("Engineering features...")
                engineer_features('data/processed/telecom_churn_train_encoded.csv', 'data/processed/telecom_churn_train_features.csv')
                st.write("Training model...")
                train_model('data/processed/telecom_churn_train_features.csv', 'models/best_model.pkl')
                st.write("Pipeline complete!")
            st.rerun()

        raw_data = pd.read_csv('data/raw/telecom_churn_train.csv')
        cleaned_data = pd.read_csv('data/processed/telecom_churn_train_cleaned.csv')
        encoded_data = pd.read_csv('data/processed/telecom_churn_train_encoded.csv')
        final_data = pd.read_csv('data/processed/telecom_churn_train_features.csv')
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Raw Data", f"{raw_data.shape[0]} rows")
        col2.metric("After Cleaning", f"{cleaned_data.shape[0]} rows")
        col3.metric("After Encoding", f"{encoded_data.shape[1]} cols")
        col4.metric("Final Features", f"{final_data.shape[1]} cols")

        with st.expander("📥 View Data Stages"):
            st.write("**Raw Data Head**")
            st.dataframe(raw_data.head(5), width="stretch")
            st.write("**Processed Data Head**")
            st.dataframe(final_data.head(5), width="stretch")

    with tab3:
        st.subheader("🔮 Upload Customer Data for Prediction")
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            upload_data = pd.read_csv(uploaded_file)
            if st.button("Predict Churn"):
                preds, proba = predict_churn_single(model, upload_data)
                results = pd.DataFrame({'Customer ID': range(len(preds)), 'Churn Prediction': ['Churn' if p == 1 else 'No Churn' for p in preds], 'Risk Score (%)': (proba * 100).round(1)})
                st.dataframe(results, width="stretch", height=400)
                st.download_button("📥 Download Predictions", results.to_csv(index=False), "predictions.csv", "text/csv")

    with tab4:
        st.subheader("⚠️ High Risk Customers")
        train_with_risk = train_data.copy()
        train_with_risk['risk_score'] = (metrics['probabilities'] * 100).round(1)
        high_risk = train_with_risk[train_with_risk['risk_score'] > 50].sort_values('risk_score', ascending=False).head(20)
        st.dataframe(high_risk[['risk_score'] + [c for c in high_risk.columns if c != 'risk_score']], width="stretch")

    with tab5:
        st.subheader("📈 Top Churn Drivers")
        feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).head(15)
        fig = px.bar(feature_importance, x='importance', y='feature', orientation='h', color='importance', color_continuous_scale='Reds')
        st.plotly_chart(fig, width="stretch")

else:
    st.warning("⚠️ Data or model not found. Please place 'telecom_churn_train.csv' in 'data/raw/' and run the pipeline from the 'Data Pipeline' tab.")
    if st.button("Initialize Pipeline"):
        st.rerun()

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | TeleChurn AI © 2024")
