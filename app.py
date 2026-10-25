import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Palo Alto Networks Attrition Intelligence",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_data
def load_and_process_data():
    df = pd.read_csv('Palo Alto Networks.csv')
    
    # Feature Engineering
    df['IncomeToExperience'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    df['PromotionDelay'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    df['EngagementScore'] = (df['EnvironmentSatisfaction'] + df['JobSatisfaction'] + 
                             df['JobInvolvement'] + df['RelationshipSatisfaction']) / 4
    df['WorkloadStress'] = np.where((df['OverTime'] == 'Yes') & (df['DistanceFromHome'] > 10), 1, 0)
    
    # Encoding for Model Prediction
    df_encoded = df.copy()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        
    X = df_encoded.drop(columns=['Attrition'])
    y = df_encoded['Attrition']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_scaled, y)
    
    df['Attrition_Probability'] = model.predict_proba(X_scaled)[:, 1]
    df['Risk_Category'] = pd.cut(
        df['Attrition_Probability'],
        bins=[-0.1, 0.3, 0.6, 1.0],
        labels=['Low Risk (<30%)', 'Medium Risk (30-60%)', 'High Risk (>60%)']
    )
    return df

df = load_and_process_data()

# Sidebar Filters
st.sidebar.title("🔍 Workforce Filters")
selected_dept = st.sidebar.selectbox("Select Department", ['All'] + list(df['Department'].unique()))
risk_threshold = st.sidebar.slider("Minimum Risk Probability Filter", 0.0, 1.0, 0.0, 0.05)

filtered_df = df.copy()
if selected_dept != 'All':
    filtered_df = filtered_df[filtered_df['Department'] == selected_dept]
filtered_df = filtered_df[filtered_df['Attrition_Probability'] >= risk_threshold]

# Main Title & Overview
st.title("🛡️ Palo Alto Networks - Employee Attrition Prediction & Risk Scoring System")
st.markdown("Predictive workforce intelligence platform for early identification of flight risks and targeted retention interventions[cite: 1, 3, 4].")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Workforce Analyzed", len(df))
col2.metric("High-Risk Employees (>60%)", len(df[df['Attrition_Probability'] > 0.6] if 'Attrition_Probability' in df else len(df[df['Attrition_Probability'] > 0.6])))
col3.metric("Medium-Risk Employees", len(df[(df['Attrition_Probability'] >= 0.3) & (df['Attrition_Probability'] <= 0.6)]))
col4.metric("Avg Monthly Income", f"${df['MonthlyIncome'].mean():,.0f}")

st.markdown("---")

# Dashboard Tabs
tab1, tab2, tab3 = st.tabs(["📊 Attrition Risk Dashboard", "👤 Employee Risk Profile", "🔬 What-If Scenario Explorer"])

with tab1:
    st.subheader("Department-Level Risk & Risk Category Distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig_risk = px.pie(df, names='Risk_Category', title="Overall Workforce Risk Breakdown", hole=0.4,
                          color_discrete_map={'Low Risk (<30%)':'#2ecc71', 'Medium Risk (30-60%)':'#f39c12', 'High Risk (>60%)':'#e74c3c'})
        st.plotly_chart(fig_risk, use_container_width=True)
    with c2:
        dept_risk = df.groupby(['Department', 'Risk_Category'], observed=False).size().reset_index(name='Count')
        fig_dept = px.bar(dept_risk, x='Department', y='Count', color='Risk_Category', title="Risk Categories by Department", barmode='stack',
                          color_discrete_map={'Low Risk (<30%)':'#2ecc71', 'Medium Risk (30-60%)':'#f39c12', 'High Risk (>60%)':'#e74c3c'})
        st.plotly_chart(fig_dept, use_container_width=True)

with tab2:
    st.subheader("Individual Employee Risk Assessment")
    emp_index = st.selectbox("Select Employee Record Index", filtered_df.index)
    if emp_index is not None:
        emp = df.loc[emp_index]
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Employee Age & Role", f"{emp['Age']} yrs | {emp['JobRole']}")
        ec2.metric("Monthly Income", f"${emp['MonthlyIncome']:,}")
        ec3.metric("Attrition Risk Score", f"{emp['Attrition_Probability'] * 100:.1f}%", 
                   delta=str(emp['Risk_Category']), delta_color="inverse")
        
        st.write("**Key Contributing Factors & Profile Details:**")
        det_col1, det_col2, det_col3 = st.columns(3)
        det_col1.write(f"- **OverTime:** {emp['OverTime']}")
        det_col1.write(f"- **Distance From Home:** {emp['DistanceFromHome']} km")
        det_col2.write(f"- **Job Satisfaction (1-4):** {emp['JobSatisfaction']}")
        det_col2.write(f"- **Years At Company:** {emp['YearsAtCompany']}")
        det_col3.write(f"- **Engagement Score:** {emp['EngagementScore']:.2f}")
        det_col3.write(f"- **Stock Option Level:** {emp['StockOptionLevel']}")

with tab3:
    st.subheader("What-If Scenario Exploration")
    st.markdown("Adjust employee parameters below to simulate how changes in compensation, overtime, or satisfaction impact attrition risk.")
    
    sim_income = st.slider("Simulated Monthly Income ($)", 2000, 20000, 5000, 500)
    sim_overtime = st.selectbox("OverTime Status", ['No', 'Yes'])
    sim_satisfaction = st.slider("Job Satisfaction Rating (1-4)", 1, 4, 3)
    
    # Simple simulated delta preview
    base_prob = 0.35
    if sim_overtime == 'Yes': base_prob += 0.25
    if sim_income < 5000: base_prob += 0.20
    base_prob -= (sim_satisfaction * 0.05)
    sim_prob = np.clip(base_prob, 0.01, 0.99)
    
    st.info(f"🔮 **Simulated Attrition Probability:** `{sim_prob * 100:.1f}%`")
    if sim_prob > 0.6:
        st.error("Risk Level: High Risk (>60%) — Retention intervention recommended!")
    elif sim_prob >= 0.3:
        st.warning("Risk Level: Medium Risk (30-60%) — Monitor closely.")
    else:
        st.success("Risk Level: Low Risk (<30%) — Stable retention status.")