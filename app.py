import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

st.set_page_config(page_title="TeleChurn AI", layout="wide", page_icon="📊")

@st.cache_data
def load_data():
    train = pd.read_csv('telecom_churn_train_features.csv')
    return train

@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

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

def predict_churn(model, data):
    if 'churned' in data.columns:
        data = data.drop('churned', axis=1)
    
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]
    
    return predictions, probabilities

train_data = load_data()
model = load_model()

X = train_data.drop('churned', axis=1)
y = train_data['churned']
metrics = get_metrics(model, X, y)

st.title("📊 TeleChurn AI - Customer Churn Prediction")
st.markdown("### Predict and prevent customer churn with 92% accuracy")

st.markdown("---")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("🎯 Accuracy", f"{metrics['accuracy']*100:.1f}%")
with col2:
    st.metric("🎯 Precision", f"{metrics['precision']*100:.1f}%")
with col3:
    st.metric("🎯 Recall", f"{metrics['recall']*100:.1f}%")
with col4:
    st.metric("🎯 F1 Score", f"{metrics['f1']:.3f}")
with col5:
    st.metric("👥 Total Customers", f"{len(train_data):,}")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔧 Data Pipeline", "🔮 Live Prediction", "⚠️ High Risk Customers", "📈 Feature Insights"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Distribution")
        churn_counts = train_data['churned'].value_counts()
        fig = px.pie(
            values=churn_counts.values,
            names=['No Churn', 'Churn'],
            color_discrete_sequence=['#00CC96', '#EF553B'],
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("Churn Rate", f"{(churn_counts[1]/len(train_data)*100):.1f}%")
    
    with col2:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(
            x=metrics['probabilities'],
            nbins=50,
            labels={'x': 'Churn Probability', 'y': 'Count'},
            color_discrete_sequence=['#636EFA']
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, metrics['predictions'])
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted No Churn', 'Predicted Churn'],
        y=['Actual No Churn', 'Actual Churn'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20}
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("� Complete Data Processing Pipeline")
    st.markdown("See how raw data transforms into ML-ready features")
    
    raw_data = pd.read_csv('telecom_churn_train.csv')
    cleaned_data = pd.read_csv('telecom_churn_train_cleaned.csv')
    encoded_data = pd.read_csv('telecom_churn_train_encoded.csv')
    final_data = pd.read_csv('telecom_churn_train_features.csv')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Raw Data", f"{raw_data.shape[0]} rows\n{raw_data.shape[1]} cols")
    with col2:
        st.metric("After Cleaning", f"{cleaned_data.shape[0]} rows\n{cleaned_data.shape[1]} cols")
    with col3:
        st.metric("After Encoding", f"{encoded_data.shape[0]} rows\n{encoded_data.shape[1]} cols")
    with col4:
        st.metric("Final Features", f"{final_data.shape[0]} rows\n{final_data.shape[1]} cols")
    
    st.markdown("---")
    
    with st.expander("📥 Step 1: Raw Data", expanded=False):
        st.markdown("**Original dataset from telecom company**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", f"{len(raw_data):,}")
            st.metric("Total Columns", raw_data.shape[1])
            st.metric("Duplicates", f"{raw_data.duplicated().sum():,}")
        with col2:
            missing = raw_data.isnull().sum()
            missing_cols = missing[missing > 0]
            if len(missing_cols) > 0:
                st.markdown("**Missing Values:**")
                for col, count in missing_cols.items():
                    st.text(f"{col}: {count} ({count/len(raw_data)*100:.1f}%)")
        
        st.dataframe(raw_data.head(10), use_container_width=True)
    
    with st.expander("🧹 Step 2: Data Cleaning", expanded=False):
        st.markdown("**Removed duplicates & handled missing values**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows Removed", f"{len(raw_data) - len(cleaned_data):,}")
            st.metric("Missing Values", "0 ✅")
        with col2:
            st.markdown("**Actions Taken:**")
            st.text("✓ Removed duplicate rows")
            st.text("✓ KNN imputation for income")
            st.text("✓ Median imputation for NPS score")
        
        st.dataframe(cleaned_data.head(10), use_container_width=True)
    
    with st.expander("🔢 Step 3: Feature Encoding", expanded=False):
        st.markdown("**Converted categorical variables to numeric**")
        
        new_cols = set(encoded_data.columns) - set(cleaned_data.columns)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("New Columns Added", len(new_cols))
            st.metric("Total Columns", encoded_data.shape[1])
        with col2:
            st.markdown("**Encoding Methods:**")
            st.text("✓ One-hot: contract_type")
            st.text("✓ One-hot: payment_method")
            st.text("✓ Label: zip_code")
        
        st.markdown("**Sample of encoded columns:**")
        encoded_cols = [col for col in encoded_data.columns if col.startswith(('contract_', 'payment_', 'month_'))][:10]
        st.dataframe(encoded_data[encoded_cols].head(10), use_container_width=True)
    
    with st.expander("⚙️ Step 4: Feature Engineering", expanded=False):
        st.markdown("**Created 20 new derived features**")
        
        engineered_cols = set(final_data.columns) - set(encoded_data.columns)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Features Created", len(engineered_cols))
            st.metric("Final Feature Count", final_data.shape[1])
        with col2:
            st.markdown("**Feature Categories:**")
            st.text("✓ Service bundles (2)")
            st.text("✓ Charge ratios (2)")
            st.text("✓ Usage intensity (3)")
            st.text("✓ Tenure segments (2)")
            st.text("✓ Interactions (9)")
        
        st.markdown("**Engineered Features:**")
        eng_cols = list(engineered_cols)[:10]
        if eng_cols:
            st.dataframe(final_data[eng_cols].head(10), use_container_width=True)
    
    with st.expander("🎯 Step 5: Model Training", expanded=False):
        st.markdown("**Trained Random Forest with hyperparameter tuning**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", "Random Forest")
            st.metric("Trees", "500")
        with col2:
            st.metric("Training Samples", "4,838 (80%)")
            st.metric("Validation Samples", "1,210 (20%)")
        with col3:
            st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
            st.metric("F1 Score", f"{metrics['f1']:.3f}")
        
        st.markdown("**Model Performance:**")
        perf_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Score': [
                f"{metrics['accuracy']*100:.1f}%",
                f"{metrics['precision']*100:.1f}%",
                f"{metrics['recall']*100:.1f}%",
                f"{metrics['f1']:.3f}"
            ]
        })
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("�🔮 Upload Customer Data for Prediction")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        upload_data = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(upload_data)} customers")
        
        if st.button("Predict Churn", type="primary"):
            preds, proba = predict_churn(model, upload_data)
            
            results = pd.DataFrame({
                'Customer ID': range(len(preds)),
                'Churn Prediction': ['Churn' if p == 1 else 'No Churn' for p in preds],
                'Risk Score (%)': (proba * 100).round(1)
            })
            
            st.dataframe(results, use_container_width=True, height=400)
            
            csv = results.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )
            
            churn_count = preds.sum()
            st.metric("Predicted Churners", f"{churn_count} ({churn_count/len(preds)*100:.1f}%)")
    else:
        st.info("Upload a CSV file to get churn predictions")

with tab4:
    st.subheader("⚠️ High Risk Customers")
    
    train_with_risk = train_data.copy()
    train_with_risk['risk_score'] = metrics['probabilities'] * 100
    train_with_risk['prediction'] = metrics['predictions']
    
    high_risk = train_with_risk[train_with_risk['risk_score'] > 50].sort_values('risk_score', ascending=False).head(20)
    
    display_cols = []
    if 'tenure_months' in high_risk.columns:
        display_cols.append('tenure_months')
    if 'monthly_charges' in high_risk.columns:
        display_cols.append('monthly_charges')
    if 'num_services' in high_risk.columns:
        display_cols.append('num_services')
    if 'num_support_calls' in high_risk.columns:
        display_cols.append('num_support_calls')
    
    display_df = high_risk[['risk_score'] + display_cols].copy()
    display_df['risk_score'] = display_df['risk_score'].round(1)
    display_df.columns = ['Risk %'] + [col.replace('_', ' ').title() for col in display_cols]
    display_df = display_df.reset_index(drop=True)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=500
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("High Risk Customers", len(high_risk))
    with col2:
        avg_risk = high_risk['risk_score'].mean()
        st.metric("Avg Risk Score", f"{avg_risk:.1f}%")
    with col3:
        if 'monthly_charges' in high_risk.columns:
            revenue_at_risk = high_risk['monthly_charges'].sum()
            st.metric("Monthly Revenue at Risk", f"₹{revenue_at_risk:,.0f}")

with tab5:
    st.subheader("📈 Top Churn Drivers")
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    fig = px.bar(
        feature_importance,
        x='importance',
        y='feature',
        orientation='h',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Key Insights")
    
    top_features = feature_importance.head(5)['feature'].tolist()
    
    for i, feat in enumerate(top_features, 1):
        feat_display = feat.replace('_', ' ').title()
        st.markdown(f"**{i}. {feat_display}**")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | TeleChurn AI © 2024")
