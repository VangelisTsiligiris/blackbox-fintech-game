import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ========================= CONFIGURATION =========================
st.set_page_config(
    page_title="Fintech Black Box Auditor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper: Synthetic Data Generator with "Baked-in" Bias
@st.cache_data
def generate_data(n_samples=1000, bias_level=0.0):
    np.random.seed(42)
    
    # 1. Generate Demographics
    ids = range(1, n_samples + 1)
    groups = np.random.choice(['Group A', 'Group B'], size=n_samples, p=[0.7, 0.3])  # ✅ Fixed: added group labels
    
    # 2. Generate Financials
    income_base = np.where(groups == 'Group A', 65000, 55000) 
    income = np.abs(np.random.normal(income_base, 15000)).astype(int)
    years_employed = np.random.randint(0, 20, size=n_samples)
    
    # 3. Generate Debt
    debt = np.abs(np.random.normal(10000, 5000)).astype(int)
    
    # 4. Financial Score
    financial_score = (income * 0.5) + (years_employed * 1000) - (debt * 1.2)
    
    # 5. Introduce Bias in Labels
    bias_penalty = np.where(groups == 'Group B', 20000 * bias_level, 0)
    final_score = financial_score - bias_penalty + np.random.normal(0, 2000, n_samples)
    
    threshold = np.percentile(final_score, 40)
    labels = (final_score > threshold).astype(int)
    
    df = pd.DataFrame({
        "Applicant_ID": ids,
        "Group": groups,
        "Income": income,
        "Debt": debt,
        "Years_Employed": years_employed,
        "Financial_Score": financial_score,
        "Historical_Outcome": labels
    })
    
    return df

# ========================= MAIN APP =========================

st.title("🕵️‍♀️ Fintech Black Box Auditor")
st.markdown("""
**Workshop Module:** Principles of Fintech & Ethical AI.  
**Objective:** Use this dashboard to audit an AI Credit Scoring algorithm, detect bias, and attempt to explain the unexplainable.
""")

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("⚙️ Model Simulation Settings")
st.sidebar.info("Adjust these sliders to simulate different market conditions and re-train the AI.")

n_applicants = st.sidebar.slider("Sample Size", 500, 5000, 1000)
bias_slider = st.sidebar.slider("Historical Data Bias Level", 0.0, 1.0, 0.8,
                                help="0=No Bias in training data. 1=Heavy discrimination against Group B in training data.")

# Generate Data
df_original = generate_data(n_samples=n_applicants, bias_level=bias_slider)

# Prepare features for model
X = df_original.drop(columns=['Applicant_ID', 'Historical_Outcome'])
y = df_original['Historical_Outcome']

# One-hot encode Group
X_encoded = pd.get_dummies(X, columns=['Group'], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Apply model to full dataset for visualization
X_full_encoded = pd.get_dummies(df_original.drop(columns=['Applicant_ID', 'Historical_Outcome']), columns=['Group'], drop_first=True)
# Ensure same columns as training data
for col in X_train.columns:
    if col not in X_full_encoded.columns:
        X_full_encoded[col] = 0
X_full_encoded = X_full_encoded[X_train.columns]  # align column order

df_with_pred = df_original.copy()
df_with_pred['AI_Prediction'] = clf.predict(X_full_encoded)
df_with_pred['AI_Label'] = df_with_pred['AI_Prediction'].map({1: "✅ Approved", 0: "❌ Rejected"})

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["Step 1: Data", "Step 2: AI Decisions", "Step 3: Fairness Audit", "Step 4: Case Study"])

with tab1:
    st.subheader("Step 1: Understand the Training Data")
    st.markdown("This data represents historical loan outcomes. The AI learns from this history. If history is biased, the AI will be biased.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Applicants", n_applicants)
        st.metric("Group A Count", len(df_with_pred[df_with_pred['Group']=='Group A']))
    with col2:
        st.metric("Overall Approval Rate", f"{df_with_pred['Historical_Outcome'].mean()*100:.1f}%")
        st.metric("Group B Count", len(df_with_pred[df_with_pred['Group']=='Group B']))

    fig = px.histogram(df_with_pred, x="Income", color="Group", barmode="overlay", title="Income Distribution by Group",
                       color_discrete_map={'Group A':'#636EFA', 'Group B':'#EF553B'})
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Raw Data Sample"):
        st.dataframe(df_with_pred.head(10))

with tab2:
    st.subheader("Step 2: The Black Box Decision")
    st.markdown(f"""
    We trained a **Random Forest** algorithm on the data.  
    **Model Accuracy:** `{accuracy*100:.1f}%`  
    Below: **Income vs. Debt**, colored by AI decision.
    """)
    
    fig_scatter = px.scatter(
        df_with_pred, 
        x="Income", 
        y="Debt", 
        color="AI_Label",
        symbol="Group",
        hover_data=["Applicant_ID", "Years_Employed", "Financial_Score"],  # ✅ Fixed: added hover_data
        color_discrete_map={"✅ Approved": "green", "❌ Rejected": "red"},
        title="AI Decision Boundary: Income vs Debt (Shape = Demographic Group)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.warning("""
    **Look closely:** Are there individuals from **Group B (Diamonds)** with high income and low debt but still **rejected**?
    This suggests the model learned bias.
    """)

with tab3:
    st.subheader("Step 3: Fairness & Ethics Audit")
    
    approve_a = df_with_pred[df_with_pred['Group'] == 'Group A']['AI_Prediction'].mean()
    approve_b = df_with_pred[df_with_pred['Group'] == 'Group B']['AI_Prediction'].mean()
    disparate_impact = approve_b / approve_a if approve_a > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Group A Approval Rate", f"{approve_a*100:.1f}%")
    col2.metric("Group B Approval Rate", f"{approve_b*100:.1f}%")
    col3.metric("Disparate Impact Ratio", f"{disparate_impact:.2f}", 
                delta="Fair > 0.8" if disparate_impact >= 0.8 else "Bias Detected", delta_color="normal" if disparate_impact >= 0.8 else "inverse")

    if disparate_impact < 0.8:
        st.error(f"⚠️ **Bias Detected:** Group B is only {disparate_impact*100:.0f}% as likely to be approved as Group A. This violates the '80% Rule'.")
    else:
        st.success("✅ **Fairness Check Passed:** The model appears to treat groups relatively equally.")

    st.markdown("### Feature Importance (Global Explainability)")
    
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': clf.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h', title="What drives the AI's decision?")
    st.plotly_chart(fig_imp, use_container_width=True)
    
    if 'Group_Group B' in importances['Feature'].values:
        imp_score = importances.loc[importances['Feature'] == 'Group_Group B', 'Importance'].values[0]
        if imp_score > 0.05:
            st.error(f"🚨 **Ethical Alert:** The model is explicitly using 'Group_Group B' (Importance: {imp_score:.2f}) — this may constitute illegal redlining.")

with tab4:
    st.subheader("Step 4: The Simulation Task")
    
    # Find high-financial-score Group B applicants rejected by AI
    median_score = df_with_pred['Financial_Score'].median()
    victims = df_with_pred[
        (df_with_pred['Group'] == 'Group B') & 
        (df_with_pred['Financial_Score'] > median_score) & 
        (df_with_pred['AI_Prediction'] == 0)
    ].sort_values(by='Financial_Score', ascending=False)
    
    if not victims.empty:
        victim = victims.iloc[0]  # ✅ Fixed: use .iloc[0] to get first row
        st.markdown(f"""
        ### Case File: Applicant #{victim['Applicant_ID']}
        **Name (Pseudonym):** Alex Chen  
        **Demographic:** Group B  
        **Income:** ${victim['Income']:,}  
        **Debt:** ${victim['Debt']:,}  
        **Years Employed:** {victim['Years_Employed']} years  
        
        ---
        **AI Decision:** ❌ REJECTED  
        **Financial Score:** {victim['Financial_Score']:,.0f} (above median)
        
        **Your Task:**  
        Alex has written to the bank demanding an explanation. Draft a letter explaining *specifically* why the AI rejected him, based *only* on the Model Logic shown in Tabs 2 & 3.
        """)
        
        explanation = st.text_area("Draft your explanation letter here:", height=200)
        
        if st.button("Submit Explanation"):
            st.info("Did you mention his 'Group'? You legally cannot. But if the model used it (see Tab 3), how do you explain the rejection truthfully *without* revealing protected attributes?")
            
    else:
        st.success("In this simulation, no obvious high-performing victims were found. Try increasing the Bias Slider to generate more unfair rejections.")

# Footer
st.markdown("---")
st.caption("Principles of Fintech | Module 4: AI Ethics & Explainability | Built with Streamlit")