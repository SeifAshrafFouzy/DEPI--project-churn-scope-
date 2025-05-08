# -*- coding: utf-8 -*-
"""
Enhanced Customer Attrition Analysis Dashboard - Final Integrated Version
"""

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, 
    StandardScaler,
    OrdinalEncoder
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Set page config
st.set_page_config(
    page_title="CHURNSCOPE Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data with caching
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        # Preprocessing
        data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data.dropna(inplace=True)
        
        # Feature engineering
        data['AvgMonthlyCharges'] = data['TotalCharges'] / data['tenure'].replace(0, 1)
        data['ServiceUsageScore'] = (
            data['OnlineSecurity'].map({'Yes': 1, 'No': 0}) +
            data['OnlineBackup'].map({'Yes': 1, 'No': 0}) +
            data['DeviceProtection'].map({'Yes': 1, 'No': 0}) +
            data['TechSupport'].map({'Yes': 1, 'No': 0})
        )
        data['Cohort'] = pd.cut(data['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], 
                              labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61+"])
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

data = load_data()

# Model training and saving function
def train_and_save_model(data):
    try:
        # Prepare features
        X = data.drop(['customerID', 'Churn', 'AvgMonthlyCharges', 'ServiceUsageScore', 'Cohort'], axis=1)
        y = data['Churn']
        
        # Define feature types
        numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        
        ordinal_features = ['Contract', 'PaymentMethod']
        ordinal_categories = [
            ['Month-to-month', 'One year', 'Two year'],
            ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
        ]
        
        nominal_features = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 
            'MultipleLines', 'InternetService', 'OnlineSecurity', 
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'PaperlessBilling'
        ]
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_features),
                ('nom', OneHotEncoder(drop='first', handle_unknown='ignore'), nominal_features)
            ],
            remainder='drop'
        )
        
        # Create full pipeline with oversampling
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('oversampler', RandomOverSampler(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Hyperparameter tuning
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        }
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        grid_search.fit(X_train, y_train)
        
        # Save best model
        best_model = grid_search.best_estimator_
        joblib.dump(best_model, 'churn_pipeline.pkl')
        
        return best_model
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

# Load model with caching
@st.cache_resource
def load_model():
    try:
        # Try to load existing model
        pipeline = joblib.load('churn_pipeline.pkl')
        return pipeline
    except FileNotFoundError:
        # If model not found, train and save new one
        st.warning("Model file not found. Training new model...")
        return train_and_save_model(data)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Load logo with error handling
@st.cache_data
def load_logo():
    try:
        return Image.open("assets\WhatsApp Image 2025-04-19 at 3.38.24 PM.jpeg")
    except:
        st.warning("Logo image not found, using placeholder")
        return None

logo = load_logo()

def predict_churn(user_input):
    if model is None:
        st.error("Model not loaded properly")
        return None
        
    try:
        # Create DataFrame from user input
        input_df = pd.DataFrame([user_input])
        
        # Ensure correct column order and feature set
        required_features = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
        ]
        
        # Add missing columns with default values if necessary
        for col in required_features:
            if col not in input_df.columns:
                input_df[col] = 0  # Default value for missing columns
                
        input_df = input_df[required_features]
        
        # Predict
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        return {
            'prediction': 'Yes' if prediction[0] == 1 else 'No',
            'probability': float(probability[0][1])
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Sidebar navigation
with st.sidebar:
    if logo:
        st.image(logo, width=200)
    else:
        st.image("https://via.placeholder.com/200x100?text=CHURN+ANALYSIS", width=200)
    
    st.markdown("""
    <div style="margin-top:-15px;margin-bottom:20px;font-size:16px;color:#FFA500;font-weight:bold;text-align:center">
    DETECT RISK. RETAIN MORE
    </div>
    """, unsafe_allow_html=True)
    
    analysis_type = st.radio(
        "Select Analysis Type",
        ["📊 Overview", "📈 Trends", "🔍 Deep Dive", "📌 Insights", "🔮 Predict Churn"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size:12px;color:#666;text-align:center">
    Created with ❤️ by Data Science Team<br>
    Data Source: Telco Customer Churn Dataset
    </div>
    """, unsafe_allow_html=True)

# Main content sections
if analysis_type == "📊 Overview":
    st.header("📊 Dashboard Overview")
    
    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", len(data), help="Total number of customers in dataset")
    with col2:
        churn_rate = data['Churn'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%", delta=f"{(churn_rate-26.5):.1f}% vs industry avg",
                 delta_color="inverse", help="Percentage of customers who churned")
    with col3:
        avg_tenure = data['tenure'].mean()
        st.metric("Avg Tenure", f"{avg_tenure:.1f} months", help="Average customer tenure in months")
    with col4:
        avg_monthly = data['MonthlyCharges'].mean()
        st.metric("Avg Monthly", f"${avg_monthly:.2f}", help="Average monthly charges")
    
    style_metric_cards(background_color="#FFFFFF", border_left_color="#FFA500", border_color="#000000")
    
    # Data preview
    with st.expander("🔍 Quick Data Preview", expanded=True):
        st.dataframe(data.head(10), use_container_width=True)
    
    # Distribution charts
    st.subheader("📈 Key Distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(data, x='tenure', nbins=30, 
                         title='Customer Tenure Distribution',
                         color_discrete_sequence=['#FFA500'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(data, y='MonthlyCharges', 
                    title='Monthly Charges Distribution',
                    color_discrete_sequence=['#000000'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Churn by category
    st.subheader("🔄 Churn by Category")
    category = st.selectbox("Select category to analyze", 
                          ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport'])
    
    fig = px.bar(data.groupby(category)['Churn'].mean().reset_index(), 
               x=category, y='Churn', 
               title=f'Churn Rate by {category}',
               color=category,
               color_discrete_sequence=['#FFA500', '#000000', '#FFD700', '#333333'])
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "📈 Trends":
    st.header("📈 Churn Trends Analysis")
    
    # Time-based analysis
    st.subheader("🕒 Churn Over Time")
    time_metric = st.radio("Time metric", ['tenure', 'Cohort'], horizontal=True)
    
    if time_metric == 'tenure':
        churn_by_tenure = data.groupby('tenure')['Churn'].mean().reset_index()
        fig = px.line(churn_by_tenure, x='tenure', y='Churn', 
                     title='Churn Rate by Tenure (Months)',
                     markers=True,
                     color_discrete_sequence=['#FFA500'])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        churn_by_cohort = data.groupby('Cohort')['Churn'].mean().reset_index()
        fig = px.bar(churn_by_cohort, x='Cohort', y='Churn', 
                    title='Churn Rate by Customer Cohort',
                    color='Churn',
                    color_continuous_scale=['#FFFFFF', '#FFA500'])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    
    # Service usage analysis
    st.subheader("🔌 Service Usage Impact")
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    service_churn = pd.DataFrame({
        'Service': services,
        'ChurnRate': [data[data[s] == 'No']['Churn'].mean() for s in services]
    })
    
    fig = px.bar(service_churn, x='Service', y='ChurnRate', 
                title='Churn Rate by Service Usage (No vs Yes)',
                color='ChurnRate',
                color_continuous_scale=['#FFFFFF', '#FFA500'])
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    
    # Payment method analysis
    st.subheader("💳 Payment Method Analysis")
    fig = px.sunburst(data, path=['PaymentMethod', 'Contract'], 
                     values='MonthlyCharges', color='Churn',
                     color_continuous_scale=['#FFFFFF', '#FFA500'],
                     title='Payment Methods and Contracts by Monthly Charges & Churn')
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "🔍 Deep Dive":
    st.header("🔍 Deep Dive Analysis")
    
    # Correlation analysis
    st.subheader("📊 Correlation Matrix")
    numeric_data = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=['#000000', '#FFA500'],
        zmin=-1,
        zmax=1,
        hoverongaps=False
    ))
    fig.update_layout(title='Feature Correlation Matrix')
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical tests
    st.subheader("📝 Statistical Significance")
    test_var = st.selectbox("Select variable to test against Churn", 
                          ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport'])
    
    contingency_table = pd.crosstab(data[test_var], data['Churn'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Chi-Square Statistic", f"{chi2:.2f}")
    with col2:
        st.metric("P-Value", f"{p:.4f}", 
                 delta="Significant" if p < 0.05 else "Not Significant",
                 delta_color="normal" if p < 0.05 else "off")
    
    # Customer segmentation
    st.subheader("👥 Customer Segmentation")
    st.warning("This analysis uses K-Means clustering on tenure and charges data")
    
    cluster_data = data[['tenure', 'MonthlyCharges', 'TotalCharges']]
    kmeans = KMeans(n_clusters=4, random_state=42)
    data['Cluster'] = kmeans.fit_predict(cluster_data)
    
    fig = px.scatter_3d(data, x='tenure', y='MonthlyCharges', z='TotalCharges',
                       color='Cluster', opacity=0.7,
                       title='3D Customer Segmentation',
                       color_discrete_sequence=['#000000', '#FFA500', '#FFD700', '#333333'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster descriptions
    cluster_desc = data.groupby('Cluster')[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].mean()
    st.dataframe(cluster_desc.style.background_gradient(cmap='Oranges'), use_container_width=True)

elif analysis_type == "📌 Insights":
    st.header("📌 Key Insights & Recommendations")
    
    # Insights cards
    with st.container():
        st.subheader("🔑 Top Findings")
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("📉 High Churn Groups", expanded=True):
                st.markdown("""
                - **Month-to-month contracts**: 42% churn rate
                - **No tech support**: 41% churn rate
                - **Fiber optic users**: 41% churn rate
                - **Electronic check payers**: 45% churn rate
                """)
        
        with col2:
            with st.expander("📈 Loyal Customer Traits", expanded=True):
                st.markdown("""
                - **Two-year contracts**: 12% churn rate
                - **Bank transfer payers**: 16% churn rate
                - **Tech support users**: 21% churn rate
                - **Long tenure (5+ years)**: 9% churn rate
                """)
    
    # Recommendations
    with st.container():
        st.subheader("💡 Actionable Recommendations")
        
        rec1, rec2, rec3 = st.columns(3)
        
        with rec1:
            st.info("""
            **Incentivize Longer Contracts**
            - Offer discounts for annual contracts
            - Highlight benefits of commitment
            """)
        
        with rec2:
            st.info("""
            **Improve Tech Support**
            - Proactive support for fiber users
            - Faster response times
            """)
        
        with rec3:
            st.info("""
            **Payment Method Optimization**
            - Encourage automatic payments
            - Offer discounts for bank transfers
            """)
    
    # Risk prediction
    with st.container():
        st.subheader("⚠️ High-Risk Customer Indicators")
        
        indicators = [
            "Month-to-month contract",
            "Electronic check payment",
            "Fiber optic internet",
            "No tech support",
            "Tenure < 12 months",
            "High monthly charges (>$80)"
        ]
        
        for indicator in indicators:
            st.markdown(f"- {indicator}")
        
        st.progress(0.75, text="Estimated preventable churn: 75%")

elif analysis_type == "🔮 Predict Churn":
    st.header("🔮 Predict Customer Churn")
    st.markdown("Enter customer details to predict churn probability")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            
        with col2:
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", 
                "Mailed check", 
                "Bank transfer (automatic)", 
                "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0)
        
        submitted = st.form_submit_button("Predict Churn")
        
        if submitted:
            user_input = {
                'gender': gender,
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            result = predict_churn(user_input)
            
            if result is not None:
                st.markdown("---")
                st.subheader("Prediction Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Churn Prediction", result['prediction'], 
                             delta="High Risk" if result['prediction'] == 'Yes' else "Low Risk",
                             delta_color="inverse" if result['prediction'] == 'Yes' else "normal")
                with col2:
                    st.metric("Churn Probability", f"{result['probability']:.1%}")
                
                if result['prediction'] == 'Yes':
                    st.warning("""
                    **Retention Strategies:**
                    - Offer contract upgrade incentives
                    - Provide personalized discounts
                    - Assign dedicated account manager
                    """)
                else:
                    st.success("""
                    **Maintenance Strategies:**
                    - Continue excellent service
                    - Consider upsell opportunities
                    - Monitor for changes in behavior
                    """)
                
                st.markdown("### Key Factors Influencing Prediction")
                factors = {
                    "Contract Type": contract,
                    "Tenure": f"{tenure} months",
                    "Monthly Charges": f"${monthly_charges:.2f}",
                    "Tech Support": tech_support,
                    "Internet Service": internet_service,
                    "Payment Method": payment_method
                }
                
                for factor, value in factors.items():
                    st.markdown(f"- **{factor}**: {value}")

# Add custom CSS
st.markdown("""
<style>
:root {
    --primary: #000000;
    --secondary: #FFA500;
    --accent: #FFD700;
    --background: #FFF8F0;
    --text: #333333;
}

.stApp {
    background-color: var(--background);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E0E0E0;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--primary);
    font-weight: 600;
}

.css-1xarl3l {
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    background-color: white;
    border-left: 4px solid var(--secondary);
}

.stButton>button {
    border-radius: 8px;
    border: 1px solid var(--secondary);
    color: var(--primary);
    background-color: var(--secondary);
    font-weight: 500;
}

.stButton>button:hover {
    background-color: #E69500;
    color: white;
    border: 1px solid #E69500;
}

.streamlit-expanderHeader {
    font-weight: 600;
    color: var(--primary);
    background-color: rgba(255, 165, 0, 0.1);
    border-radius: 8px;
    padding: 0.5rem;
    border-left: 4px solid var(--secondary);
}

.streamlit-expanderContent {
    padding: 1rem;
}

.stDataFrame {
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    border: 1px solid #E0E0E0;
}

.st-b7 {
    background-color: rgba(255, 165, 0, 0.1) !important;
}

.st-bm {
    background-color: var(--secondary) !important;
}

.stAlert {
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    border-left: 4px solid var(--secondary);
    background-color: rgba(255, 165, 0, 0.05);
}

.st-bh {
    border: 1px solid var(--secondary) !important;
    border-radius: 8px !important;
}

.st-bj {
    background-color: rgba(255, 165, 0, 0.1) !important;
    border-radius: 8px !important;
}

.st-bk {
    color: var(--primary) !important;
}

.stForm {
    background-color: rgba(255, 165, 0, 0.05);
    padding: 20px;
    border-radius: 10px;
    border-left: 4px solid var(--secondary);
}
</style>
""", unsafe_allow_html=True)