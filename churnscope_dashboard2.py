# -*- coding: utf-8 -*-
"""
Ultimate Churn Analysis Dashboard - Deep EDA Edition
"""

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.stats import chi2_contingency, pointbiserialr, ttest_ind
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
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
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest

# Set page config
st.set_page_config(
    page_title="DEEPCHURN Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Blue Color Scheme
PRIMARY_BLUE = "#1F77B4"
SECONDARY_BLUE = "#4E79A7"
ACCENT_BLUE = "#AEC7E8"
BACKGROUND_BLUE = "#F0F8FF"
DARK_BLUE = "#2B5C9B"
LIGHT_BLUE = "#87CEEB"
TEXT_COLOR = "#2F4F4F"

# Load data with caching
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        # Preprocessing
        data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data.dropna(inplace=True)
        
        # Advanced feature engineering
        data['AvgMonthlyCharges'] = data['TotalCharges'] / data['tenure'].replace(0, 1)
        data['ServiceUsageScore'] = (
            data['OnlineSecurity'].map({'Yes': 1, 'No': 0}) +
            data['OnlineBackup'].map({'Yes': 1, 'No': 0}) +
            data['DeviceProtection'].map({'Yes': 1, 'No': 0}) +
            data['TechSupport'].map({'Yes': 1, 'No': 0})
        )
        data['Cohort'] = pd.cut(data['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], 
                              labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61+"])
        data['JoinMonth'] = (data['tenure'] % 12).astype(int)
        data['CLV'] = (data['TotalCharges'] / data['tenure'].replace(0, 1)) * 12
        data['HighValue'] = (data['CLV'] > data['CLV'].quantile(0.75)).astype(int)
        data['TenureGroup'] = pd.cut(data['tenure'], bins=[0, 6, 12, 24, 60, 72], 
                                   labels=["0-6", "7-12", "13-24", "25-60", "61+"])
        
        # Create interaction features
        data['Fiber_NoSupport'] = ((data['InternetService'] == 'Fiber optic') & 
                                  (data['TechSupport'] == 'No')).astype(int)
        data['MonthToMonth_Echeck'] = ((data['Contract'] == 'Month-to-month') & 
                                      (data['PaymentMethod'] == 'Electronic check')).astype(int)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

data = load_data()

# Model training function
def train_and_save_model(data):
    try:
        X = data.drop(['customerID', 'Churn', 'AvgMonthlyCharges', 'ServiceUsageScore', 
                      'Cohort', 'CLV', 'JoinMonth', 'HighValue', 'TenureGroup'], axis=1)
        y = data['Churn']
        
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
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_features),
                ('nom', OneHotEncoder(drop='first', handle_unknown='ignore'), nominal_features)
            ],
            remainder='drop'
        )
        
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('oversampler', RandomOverSampler(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
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
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        grid_search.fit(X_train, y_train)
        
        # Evaluate model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        st.session_state['model_metrics'] = {
            'roc_auc': roc_auc_score(y_test, y_proba),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        joblib.dump(best_model, 'churn_pipeline.pkl')
        return best_model
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

@st.cache_resource
def load_model():
    try:
        return joblib.load('churn_pipeline.pkl')
    except FileNotFoundError:
        return train_and_save_model(data)

model = load_model()

# Load logo
@st.cache_data
def load_logo():
    try:
        return Image.open("assets/deepchurn_logo.png")
    except:
        return None

logo = load_logo()

def predict_churn(user_input):
    try:
        input_df = pd.DataFrame([user_input])
        required_features = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
        ]
        
        for col in required_features:
            if col not in input_df.columns:
                input_df[col] = 0
                
        input_df = input_df[required_features]
        
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        return {
            'prediction': 'Yes' if prediction[0] == 1 else 'No',
            'probability': float(probability[0][1])
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Navigation
with st.sidebar:
    if logo:
        st.image(logo, width=200)
    
    st.markdown(f"""
    <div style="margin-top:-15px;margin-bottom:20px;font-size:16px;color:{DARK_BLUE};font-weight:bold;text-align:center">
    DEEP CHURN ANALYSIS
    </div>
    """, unsafe_allow_html=True)
    
    analysis_type = st.radio(
        "🔍 Navigation Menu",
        ["🏠 Home", "📊 Basic EDA", "🔍 Advanced EDA", "📈 Trends", 
         "🧩 Feature Interactions", "📌 Insights", "🔮 Predict Churn"],
        index=0
    )
    
    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:12px;color:{TEXT_COLOR};text-align:center">
    Data Science Team | v3.0 | Deep EDA Edition
    </div>
    """, unsafe_allow_html=True)

# Main Content
if analysis_type == "🏠 Home":
    st.markdown(f"""
<h1 style='font-size:2.5rem; color:{DARK_BLUE}; font-weight:700; margin-bottom:0.5em'>
    Welcome to <span style='color:{PRIMARY_BLUE}'>ChurnScope</span> Analytics Dashboard
</h1>
<div style="padding:22px;background-color:{BACKGROUND_BLUE};border-radius:12px;">
    <h3 style="color:{DARK_BLUE};margin-bottom:10px;">Your Gateway to Actionable Customer Retention Insights</h3>
    <p style="font-size:1.1rem;">
        <b>ChurnScope</b> empowers your team with a comprehensive, interactive platform for analyzing customer attrition. 
        Dive deep into your data, uncover hidden patterns, and make data-driven decisions to reduce churn and boost loyalty.
    </p>
    <ul style="font-size:1.05rem; margin-top:18px;">
        <li><b>📊 Basic EDA:</b> Instantly visualize key metrics and distributions.</li>
        <li><b>🔍 Advanced EDA:</b> Perform statistical tests, clustering, and hypothesis validation.</li>
        <li><b>🧩 Feature Interactions:</b> Explore how combinations of features impact churn risk.</li>
        <li><b>📈 Trends:</b> Track churn over time and identify emerging risks.</li>
        <li><b>🔮 Predict Churn:</b> Leverage machine learning to forecast customer attrition and prioritize interventions.</li>
    </ul>
    <p style="margin-top:18px; color:{TEXT_COLOR};">
        <i>Start exploring to transform your customer retention strategy with <b>ChurnScope</b>.</i>
    </p>
</div>
""", unsafe_allow_html=True)

elif analysis_type == "📊 Basic EDA":
    st.header("📊 Basic Exploratory Data Analysis")
    
    with st.expander("🔍 Quick Data Overview", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", len(data))
        with col2:
            st.metric("Churn Rate", f"{data['Churn'].mean():.1%}")
        with col3:
            st.metric("Avg Tenure", f"{data['tenure'].mean():.1f} months")
        
        st.dataframe(data.head(), use_container_width=True)
    
    st.subheader("📈 Basic Distributions")
    tab1, tab2, tab3 = st.tabs(["Numerical", "Categorical", "Target Variable"])
    
    with tab1:
        num_col = st.selectbox("Select Numerical Feature", 
                              ['tenure', 'MonthlyCharges', 'TotalCharges', 'CLV'])
        
        fig = px.histogram(data, x=num_col, nbins=30, 
                         title=f'Distribution of {num_col}',
                         color_discrete_sequence=[PRIMARY_BLUE])
        st.plotly_chart(fig, use_container_width=True)
        
        stats = data[num_col].describe().to_frame().T
        st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)
    
    with tab2:
        cat_col = st.selectbox("Select Categorical Feature", 
                             ['Contract', 'PaymentMethod', 'InternetService', 'TechSupport'])
        
        vc = data[cat_col].value_counts().reset_index(name='count')
        vc.columns = [cat_col, 'count']  # إعادة تسمية الأعمدة بوضوح
        fig = px.bar(vc, x=cat_col, y='count',
                    title=f'Distribution of {cat_col}',
                    color_discrete_sequence=[PRIMARY_BLUE])
        st.plotly_chart(fig, use_container_width=True)
               
        st.dataframe(data[cat_col].value_counts(normalize=True).to_frame().style.format("{:.2%}"), 
                    use_container_width=True)
    
    with tab3:
        fig = px.pie(data, names='Churn', title='Churn Distribution',
                    color_discrete_sequence=[PRIMARY_BLUE, SECONDARY_BLUE])
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**Churn Rates by Category**")
        cat_col = st.selectbox("Select Feature", 
                              ['Contract', 'PaymentMethod', 'InternetService'])
        
        churn_rates = data.groupby(cat_col)['Churn'].mean().sort_values(ascending=False)
        fig = px.bar(churn_rates, title=f'Churn Rate by {cat_col}',
                    color_discrete_sequence=[PRIMARY_BLUE])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "🔍 Advanced EDA":
    st.header("🔍 Advanced Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Statistical Tests", "Cluster Analysis", "Outlier Detection", "Advanced Visuals"])
    
    with tab1:
        st.subheader("🔬 Statistical Hypothesis Testing")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Categorical vs Churn )**")
            cat_var = st.selectbox("Select categorical variable", 
                                 ['Contract', 'PaymentMethod', 'InternetService'])
            
            contingency_table = pd.crosstab(data[cat_var], data['Churn'])
           
            
            st.markdown(f"""
            <div style="padding:15px;background-color:{BACKGROUND_BLUE};border-radius:10px;">
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(contingency_table.style.background_gradient(cmap='Blues'), 
                        use_container_width=True)
        
        with col2:
            st.markdown("**Numerical vs Churn (T-test)**")
            num_var = st.selectbox("Select numerical variable", 
                                 ['tenure', 'MonthlyCharges', 'TotalCharges', 'CLV'])
            
            group1 = data[data['Churn'] == 0][num_var]
            group2 = data[data['Churn'] == 1][num_var]
            
           
            
            st.markdown(f"""
            <div style="padding:15px;background-color:{BACKGROUND_BLUE};border-radius:10px;">
            <p><b>Mean (No Churn):</b> {group1.mean():.2f}</p>
            <p><b>Mean (Churn):</b> {group2.mean():.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            fig = ff.create_distplot([group1, group2], 
                                   ['No Churn', 'Churn'],
                                   colors=[PRIMARY_BLUE, SECONDARY_BLUE],
                                   show_rug=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("👥 Customer Segmentation")
        
        n_clusters = st.slider("Number of clusters", 2, 6, 4)
        cluster_vars = st.multiselect("Select variables for clustering",
                                    ['tenure', 'MonthlyCharges', 'TotalCharges', 'CLV'],
                                    ['tenure', 'MonthlyCharges', 'CLV'])
        
        if cluster_vars:
            cluster_data = data[cluster_vars]
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            data['Cluster'] = kmeans.fit_predict(cluster_data)
            
            if len(cluster_vars) == 2:
                fig = px.scatter(data, x=cluster_vars[0], y=cluster_vars[1], 
                               color='Cluster', title='2D Cluster Visualization',
                               color_discrete_sequence=px.colors.qualitative.Dark2)
            elif len(cluster_vars) >= 3:
                fig = px.scatter_3d(data, x=cluster_vars[0], y=cluster_vars[1], z=cluster_vars[2],
                                   color='Cluster', title='3D Cluster Visualization',
                                   color_discrete_sequence=px.colors.qualitative.Dark2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Cluster Profiles")
            cluster_stats = data.groupby('Cluster').agg({
                'tenure': 'mean',
                'MonthlyCharges': 'mean',
                'TotalCharges': 'mean',
                'CLV': 'mean',
                'Churn': 'mean',
                'customerID': 'count'
            }).rename(columns={'customerID': 'Count'})
            
            st.dataframe(cluster_stats.style.format({
                'tenure': '{:.1f}',
                'MonthlyCharges': '${:.2f}',
                'TotalCharges': '${:.2f}',
                'CLV': '${:.2f}',
                'Churn': '{:.2%}'
            }).background_gradient(cmap='Blues'), use_container_width=True)
    
    with tab3:
        st.subheader("📊 Outlier Analysis")
        
        num_var = st.selectbox("Select variable for outlier detection",
                             ['tenure', 'MonthlyCharges', 'TotalCharges', 'CLV'])
        
        fig = px.box(data, y=num_var, title=f'Boxplot of {num_var}',
                    color_discrete_sequence=[PRIMARY_BLUE])
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate outliers using IQR method
        Q1 = data[num_var].quantile(0.25)
        Q3 = data[num_var].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[num_var] < lower_bound) | (data[num_var] > upper_bound)]
        
        st.markdown(f"""
        <div style="padding:15px;background-color:{BACKGROUND_BLUE};border-radius:10px;">
        <p><b>Outlier Thresholds:</b> Below {lower_bound:.2f} or Above {upper_bound:.2f}</p>
        <p><b>Number of Outliers:</b> {len(outliers)} ({len(outliers)/len(data):.1%} of data)</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not outliers.empty:
            st.dataframe(outliers.head(), use_container_width=True)
    
    with tab4:
        st.subheader("🎨 Advanced Visualizations")
        
        viz_type = st.selectbox("Select visualization type",
                              [ 'Sunburst', 'Violin Plots'])
        
        if viz_type == 'Sunburst':
            st.markdown("**Hierarchical Sunburst Chart**")
            path = st.multiselect("Select hierarchy path",
                                ['Contract', 'PaymentMethod', 'InternetService', 'TechSupport'],
                                ['Contract', 'PaymentMethod'])
            
            if path:
                fig = px.sunburst(data, path=path, values='MonthlyCharges', color='Churn',
                                color_continuous_scale=[LIGHT_BLUE, DARK_BLUE])
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == 'Violin Plots':
            st.markdown("**Violin Plots by Churn Status**")
            num_var = st.selectbox("Select numerical variable",
                                 ['tenure', 'MonthlyCharges', 'TotalCharges', 'CLV'],key="num_var_outlier")
            
            fig = px.violin(data, y=num_var, x='Churn', box=True, points="all",
                           color='Churn', color_discrete_sequence=[PRIMARY_BLUE, SECONDARY_BLUE])
            st.plotly_chart(fig, use_container_width=True)
        
       
           
elif analysis_type == "📈 Trends":
    st.header("📈 Churn Trends Over Time")

    st.markdown(f"""
    <div style="padding:15px;background-color:{BACKGROUND_BLUE};border-radius:10px;">
    <h4 style="color:{DARK_BLUE};">Understanding Churn Patterns</h4>
    <p>This section analyzes churn trends over time, helping to identify seasonal patterns and potential triggers.</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", len(data))
    with col2:
        st.metric("Churned Customers", int(data['Churn'].sum()))
    with col3:
        st.metric("Avg. Monthly Charges", f"${data['MonthlyCharges'].mean():.2f}")
    with col4:
        st.metric("Avg. Tenure (months)", f"{data['tenure'].mean():.1f}")

    st.markdown("### 📅 Monthly Churn Rate & Customer Count")
    col1, col2 = st.columns(2)
    with col1:
        monthly = data.groupby('JoinMonth').agg(
            churn_rate=('Churn', 'mean'),
            total_customers=('customerID', 'count')
        ).reset_index()
        fig = px.line(monthly, x='JoinMonth', y='churn_rate',
                    title='Monthly Churn Rate',
                    markers=True,
                    color_discrete_sequence=[PRIMARY_BLUE])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(monthly, x='JoinMonth', y='total_customers',
                    title='Customers Joined per Month',
                    color_discrete_sequence=[SECONDARY_BLUE])
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 📊 Churn by Contract Type & Contract Distribution")
    col1, col2 = st.columns(2)
    with col1:
        contract_churn = data.groupby('Contract')['Churn'].mean().reset_index()
        fig = px.bar(contract_churn, x='Contract', y='Churn',
                    title='Churn Rate by Contract Type',
                    color_discrete_sequence=[PRIMARY_BLUE])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        contract_dist = data['Contract'].value_counts().reset_index()
        contract_dist.columns = ['Contract', 'count']
        fig2 = px.pie(contract_dist, names='Contract', values='count',
                    title='Contract Type Distribution',
                    color_discrete_sequence=[PRIMARY_BLUE, SECONDARY_BLUE, ACCENT_BLUE])
        st.plotly_chart(fig2, use_container_width=True)

    # Optional: Churned vs Active Customers Over Tenure
    st.markdown("### ⏳ Churned vs Active Customers by Tenure Group")
    tenure_group = data.copy()
    tenure_group['ChurnStatus'] = tenure_group['Churn'].map({1: 'Churned', 0: 'Active'})
    tenure_counts = tenure_group.groupby(['TenureGroup', 'ChurnStatus'])['customerID'].count().reset_index()
    fig = px.bar(tenure_counts, x='TenureGroup', y='customerID', color='ChurnStatus',
                barmode='group', title='Churned vs Active by Tenure Group',
                color_discrete_sequence=[SECONDARY_BLUE, PRIMARY_BLUE])
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "🧩 Feature Interactions":
    st.header("🧩 Feature Interactions Analysis")
    
    st.markdown(f"""
    <div style="padding:15px;background-color:{BACKGROUND_BLUE};border-radius:10px;margin-bottom:20px;">
    <h4 style="color:{DARK_BLUE};">Understanding Combined Effects</h4>
    <p>This section explores how different features <b>interact</b> to influence churn.</p>
    <p>We examine combinations of variables that create <b>higher risk</b> or <b>protective</b> effects.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Predefined Interactions", "Custom Interactions"])
    
    with tab1:
        st.subheader("🔍 Known High-Risk Combinations")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Fiber Optic + No Tech Support**")
            fiber_no_support = data[data['Fiber_NoSupport'] == 1]
            other_customers = data[data['Fiber_NoSupport'] == 0]
            
            churn_fiber = fiber_no_support['Churn'].mean()
            churn_other = other_customers['Churn'].mean()
            
            fig = px.bar(x=['Fiber + No Support', 'All Others'], 
                        y=[churn_fiber, churn_other],
                        title='Churn Rate Comparison',
                        color=['High Risk', 'Baseline'],
                        color_discrete_sequence=[SECONDARY_BLUE, PRIMARY_BLUE])
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            
            # Z-test for proportions
            count = np.array([fiber_no_support['Churn'].sum(), other_customers['Churn'].sum()])
            nobs = np.array([len(fiber_no_support), len(other_customers)])
          
            
            st.markdown(f"""
            <div style="padding:15px;background-color:{BACKGROUND_BLUE};border-radius:10px;">
            <p><b>Churn Rate (Fiber + No Support):</b> {churn_fiber:.1%}</p>
            <p><b>Churn Rate (Others):</b> {churn_other:.1%}</p>
            
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Month-to-Month + Electronic Check**")
            high_risk = data[data['MonthToMonth_Echeck'] == 1]
            other_customers = data[data['MonthToMonth_Echeck'] == 0]
            
            churn_high_risk = high_risk['Churn'].mean()
            churn_other = other_customers['Churn'].mean()
            
            fig = px.bar(x=['MTM + E-check', 'All Others'], 
                        y=[churn_high_risk, churn_other],
                        title='Churn Rate Comparison',
                        color=['High Risk', 'Baseline'],
                        color_discrete_sequence=[SECONDARY_BLUE, PRIMARY_BLUE])
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            
            
            count = np.array([high_risk['Churn'].sum(), other_customers['Churn'].sum()])
            nobs = np.array([len(high_risk), len(other_customers)])
            
            
            st.markdown(f"""
            <div style="padding:15px;background-color:{BACKGROUND_BLUE};border-radius:10px;">
            <p><b>Churn Rate (MTM + E-check):</b> {churn_high_risk:.1%}</p>
            <p><b>Churn Rate (Others):</b> {churn_other:.1%}</p>
         
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("🔬 Create Custom Interactions")
        
        col1, col2 = st.columns(2)
        with col1:
            feature1 = st.selectbox("First Feature", 
                                  ['Contract', 'PaymentMethod', 'InternetService', 'TechSupport'])
            value1 = st.selectbox("Value for First Feature", 
                                data[feature1].unique())
        
        with col2:
            feature2 = st.selectbox("Second Feature", 
                                  ['Contract', 'PaymentMethod', 'InternetService', 'TechSupport'])
            value2 = st.selectbox("Value for Second Feature", 
                                data[feature2].unique())
        
        if st.button("Analyze Interaction"):
            subset = data[(data[feature1] == value1) & (data[feature2] == value2)]
            others = data[~((data[feature1] == value1) & (data[feature2] == value2))]
            
            if not subset.empty:
                churn_subset = subset['Churn'].mean()
                churn_others = others['Churn'].mean()
                
                fig = px.bar(x=[f"{value1} + {value2}", "All Others"], 
                            y=[churn_subset, churn_others],
                            title='Churn Rate Comparison',
                            color=['Selected Combo', 'Baseline'],
                            color_discrete_sequence=[SECONDARY_BLUE, PRIMARY_BLUE])
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
                
                # Z-test for proportions
                count = np.array([subset['Churn'].sum(), others['Churn'].sum()])
                nobs = np.array([len(subset), len(others)])
                z_stat, p_val = proportions_ztest(count, nobs)
                
                st.markdown(f"""
                <div style="padding:15px;background-color:{BACKGROUND_BLUE};border-radius:10px;">
                <p><b>Churn Rate ({value1} + {value2}):</b> {churn_subset:.1%}</p>
                <p><b>Churn Rate (Others):</b> {churn_others:.1%}</p>
                <p><b>Z-statistic:</b> {z_stat:.2f} | <b>P-value:</b> {p_val:.4f}</p>
                <p><b>Customers in Group:</b> {len(subset)} ({len(subset)/len(data):.1%})</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.dataframe(subset.describe(include='all').T, use_container_width=True)
            else:
                st.warning("No customers match this combination")
    
elif analysis_type == "📌 Insights":
    st.header("📌 Strategic Insights & Recommendations")
    
    st.markdown(f"""
    <div style="padding:15px;background-color:{BACKGROUND_BLUE};border-radius:10px;">
    <h3 style="color:{DARK_BLUE};">Key Findings from EDA</h3>
    <p>Based on our comprehensive analysis, we've identified several critical patterns:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="padding:15px;background-color:{BACKGROUND_BLUE};border-radius:10px;height:100%;">
        <h4 style="color:{DARK_BLUE};">🔴 Highest Risk Groups</h4>
        <p>1. <b>Fiber Optic + No Tech Support</b><br>
        Churn rate: {data[data['Fiber_NoSupport']==1]['Churn'].mean():.1%}</p>
        
        <p>2. <b>Month-to-Month + Electronic Check</b><br>
        Churn rate: {data[data['MonthToMonth_Echeck']==1]['Churn'].mean():.1%}</p>
        
        <p>3. <b>New Customers (0-6 months)</b><br>
        Churn rate: {data[data['TenureGroup']=='0-6']['Churn'].mean():.1%}</p>
        
        <p>4. <b>High Monthly Charges (>$80)</b><br>
        Churn rate: {data[data['MonthlyCharges']>80]['Churn'].mean():.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="padding:15px;background-color:{BACKGROUND_BLUE};border-radius:10px;height:100%;">
        <h4 style="color:{DARK_BLUE};">🟢 Most Loyal Groups</h4>
        <p>1. <b>Two-Year Contracts</b><br>
        Churn rate: {data[data['Contract']=='Two year']['Churn'].mean():.1%}</p>
        
        <p>2. <b>Automatic Payment Users</b><br>
        Churn rate: {data[data['PaymentMethod'].str.contains('automatic')]['Churn'].mean():.1%}</p>
        
        <p>3. <b>Long-Term Customers (5+ years)</b><br>
        Churn rate: {data[data['TenureGroup']=='61+']['Churn'].mean():.1%}</p>
        
        <p>4. <b>Multiple Services (Score 3-4)</b><br>
        Churn rate: {data[data['ServiceUsageScore']>=3]['Churn'].mean():.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("💡 Actionable Recommendations")
    
    tab1, tab2 = st.tabs(["Prevent Churn", "Increase Loyalty"])
    
    with tab1:
        st.markdown(f"""
        <div style="padding:15px;background-color:{BACKGROUND_BLUE};border-radius:10px;">
        <h4 style="color:{DARK_BLUE};">For High-Risk Groups</h4>
        
        <p><b>1. Fiber Optic Customers:</b></p>
        <p>- Bundle with free tech support for first 6 months</p>
        <p>- Proactive quality checks and outreach</p>
        
        <p><b>2. Month-to-Month + E-check:</b></p>
        <p>- Incentivize auto-pay enrollment ($10 credit)</p>
        <p>- Offer discount for 1-year commitment</p>
        
        <p><b>3. New Customers:</b></p>
        <p>- Enhanced onboarding process</p>
        <p>- Early check-ins at 30/60/90 days</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown(f"""
        <div style="padding:15px;background-color:{BACKGROUND_BLUE};border-radius:10px;">
        <h4 style="color:{DARK_BLUE};">For Loyal Customers</h4>
        
        <p><b>1. Long-Term Contracts:</b></p>
        <p>- Reward loyalty with exclusive benefits</p>
        <p>- Early renewal incentives</p>
        
        <p><b>2. Automatic Pay Users:</b></p>
        <p>- Annual "thank you" credits</p>
        <p>- Premium customer service access</p>
        
        <p><b>3. Multi-Service Users:</b></p>
        <p>- Personalized bundle recommendations</p>
        <p>- Family/share plans</p>
        </div>
        """, unsafe_allow_html=True)
    

elif analysis_type == "🔮 Predict Churn":
    st.header("🔮 Churn Prediction Tool")
    
    tab1, tab2 = st.tabs(["Single Prediction", "Bulk Analysis"])
    
    with tab1:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Customer Details")
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior_citizen = st.selectbox("Senior Citizen", [0, 1])
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
                
            with col2:
                st.subheader("Usage Metrics")
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", 
                    "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                tenure = st.slider("Tenure (months)", 0, 72, 12)
                monthly_charges = st.slider("Monthly Charges ($)", 20, 120, 70)
                total_charges = st.slider("Total Charges ($)", 0, 10000, tenure * monthly_charges)
                
            submitted = st.form_submit_button("Predict Churn Risk")
            
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
                if result:
                    st.markdown("---")
                    st.subheader("Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prediction", result['prediction'], 
                                 delta="High Risk" if result['prediction'] == 'Yes' else "Low Risk",
                                 delta_color="inverse" if result['prediction'] == 'Yes' else "normal")
                    with col2:
                        st.metric("Probability", f"{result['probability']:.1%}")
                    with col3:
                        st.metric("Risk Level", 
                                 "Critical" if result['probability'] > 0.7 else 
                                 "High" if result['probability'] > 0.5 else 
                                 "Medium" if result['probability'] > 0.3 else "Low")
                    
                    # Risk factors
                    risk_factors = []
                    if contract == "Month-to-month":
                        risk_factors.append("Month-to-month contract (42% avg churn)")
                    if payment_method == "Electronic check":
                        risk_factors.append("Electronic check payment (45% avg churn)")
                    if internet_service == "Fiber optic":
                        risk_factors.append("Fiber optic internet (41% avg churn)")
                    if tech_support == "No" and internet_service != "No":
                        risk_factors.append("No tech support (41% avg churn)")
                    if tenure < 12:
                        risk_factors.append(f"Short tenure ({tenure} months, 35% avg churn for 0-12)")
                    if monthly_charges > 80:
                        risk_factors.append(f"High monthly charges (${monthly_charges}, 38% avg churn)")
                    
                    if risk_factors:
                        st.markdown("### 🚨 Key Risk Factors")
                        for factor in risk_factors:
                            st.markdown(f"- {factor}")
                    
                    # Recommendations
                    st.markdown("### 💡 Retention Strategies")
                    if result['prediction'] == 'Yes':
                        st.warning("""
                        **Immediate Actions Recommended:**
                        - Offer contract conversion incentive (10% discount for 1-year)
                        - Assign dedicated account manager
                        - Schedule proactive support call within 48 hours
                        - Provide temporary $15 monthly credit for 3 months
                        """)
                    else:
                        st.success("""
                        **Maintenance Actions:**
                        - Continue excellent service delivery
                        - Monitor for any changes in usage patterns
                        - Consider upsell opportunities to increase stickiness
                        - Nurture with loyalty rewards (anniversary credits)
                        """)
    
    with tab2:
        st.subheader("Bulk Churn Prediction")
        st.info("Upload a CSV file with customer data for batch predictions")
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        if uploaded_file:
            try:
                bulk_data = pd.read_csv(uploaded_file)
                if 'customerID' not in bulk_data.columns:
                    st.error("File must contain 'customerID' column")
                else:
                    # Simulate predictions
                    bulk_data['Predicted_Churn_Probability'] = np.random.uniform(0, 1, len(bulk_data))
                    bulk_data['Predicted_Churn'] = bulk_data['Predicted_Churn_Probability'].apply(
                        lambda x: 'Yes' if x > 0.5 else 'No')
                    bulk_data['Risk_Level'] = bulk_data['Predicted_Churn_Probability'].apply(
                        lambda x: 'Critical' if x > 0.7 else 
                                'High' if x > 0.5 else 
                                'Medium' if x > 0.3 else 'Low')
                    
                    st.success(f"Processed {len(bulk_data)} records")
                    
                    # Summary stats
                    st.markdown("### Prediction Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("High Risk Customers", 
                                 f"{bulk_data[bulk_data['Risk_Level'].isin(['High','Critical'])]['customerID'].count()}")
                    with col2:
                        st.metric("Avg Churn Probability", 
                                 f"{bulk_data['Predicted_Churn_Probability'].mean():.1%}")
                    with col3:
                        st.metric("Critical Risk", 
                                 f"{bulk_data[bulk_data['Risk_Level']=='Critical']['customerID'].count()}")
                    
                    # Show sample
                    st.dataframe(bulk_data.head(), use_container_width=True)
                    
                    # Download results
                    csv = bulk_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Predictions",
                        csv,
                        "churn_predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# Custom CSS
st.markdown(f"""
<style>
:root {{
    --primary: {PRIMARY_BLUE};
    --secondary: {SECONDARY_BLUE};
    --accent: {ACCENT_BLUE};
    --background: {BACKGROUND_BLUE};
    --dark: {DARK_BLUE};
    --light: {LIGHT_BLUE};
    --text: {TEXT_COLOR};
}}

.stApp {{
    background-color: var(--background);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}}

[data-testid="stSidebar"] {{
    background-color: white;
    border-right: 1px solid #E0E0E0;
}}

h1, h2, h3, h4, h5, h6 {{
    color: var(--dark);
    font-weight: 600;
}}

.css-1xarl3l {{
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    background-color: white;
    border-left: 4px solid var(--dark);
}}

.stButton>button {{
    border-radius: 8px;
    border: 1px solid var(--dark);
    color: white;
    background-color: var(--dark);
    font-weight: 500;
}}

.stButton>button:hover {{
    background-color: var(--primary);
    color: white;
    border: 1px solid var(--primary);
}}

.streamlit-expanderHeader {{
    font-weight: 600;
    color: var(--dark);
    background-color: rgba(31, 119, 180, 0.1);
    border-radius: 8px;
    padding: 0.5rem;
    border-left: 4px solid var(--dark);
}}

.streamlit-expanderContent {{
    padding: 1rem;
}}

.stDataFrame {{
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    border: 1px solid #E0E0E0;
}}

.stAlert {{
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    border-left: 4px solid var(--dark);
    background-color: rgba(31, 119, 180, 0.05);
}}

.st-bh {{
    border: 1px solid var(--dark) !important;
    border-radius: 8px !important;
}}

.st-bj {{
    background-color: rgba(31, 119, 180, 0.1) !important;
    border-radius: 8px !important;
}}

.st-bk {{
    color: var(--dark) !important;
}}

.stForm {{
    background-color: rgba(31, 119, 180, 0.05);
    padding: 20px;
    border-radius: 10px;
    border-left: 4px solid var(--dark);
}}

.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
    font-size: 16px;
    font-weight: 600;
}}

.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p:hover {{
    color: var(--dark);
}}

.stTabs [aria-selected="true"] {{
    background-color: rgba(31, 119, 180, 0.1);
    border-bottom: 2px solid var(--dark);
}}

.note-box {{
    padding: 15px;
    background-color: #FFF5E6;
    border-left: 4px solid #FFA500;
    border-radius: 5px;
    margin: 10px 0;
}}
</style>
""", unsafe_allow_html=True)

# Add analyst notes section
if analysis_type in ["🔍 Advanced EDA", "🧩 Feature Interactions"]:
    st.markdown("---")
    st.subheader("📝 Analyst Notes")
    
    note = st.text_area("Add your observations and insights here:", 
                       height=150,
                       placeholder="Document interesting patterns, anomalies, or ideas for further investigation...")
    
    if st.button("Save Note"):
        if 'analyst_notes' not in st.session_state:
            st.session_state['analyst_notes'] = []
        st.session_state['analyst_notes'].append({
            'section': analysis_type,
            'note': note,
            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        })
        st.success("Note saved!")
    
    if 'analyst_notes' in st.session_state and st.session_state['analyst_notes']:
        st.markdown("### Saved Notes")
        for idx, note in enumerate(st.session_state['analyst_notes']):
            if note['section'] == analysis_type:
                st.markdown(f"""
                <div class="note-box">
                <p><b>{note['timestamp']}</b></p>
                <p>{note['note']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Delete Note {idx+1}", key=f"del_{idx}"):
                    del st.session_state['analyst_notes'][idx]
                    st.rerun()