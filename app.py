import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    """
    Generates a synthetic dataset representing loan applicants.
    bias_level (0.0 to 1.0): 
    - 0.0 = Fair world (Outcome depends only on financial merit).
    - 1.0 = Biased world (Outcome heavily penalized by 'Group B' status).
    """
    np.random.seed(42)
    
    # 1. Generate Demographics
    ids = range(1, n_samples + 1)
    groups = np.random.choice(, size=n_samples, p=[0.7, 0.3])
    
    # 2. Generate Financials (Correlated slightly with group to simulate systemic inequality)
    income_base = np.where(groups == 'Group A', 65000, 55000) 
    income = np.abs(np.random.normal(income_base, 15000)).astype(int)
    
    years_employed = np.random.randint(0, 20, size=n_samples)
    
    # 3. Generate Debt (Uncorrelated)
    debt = np.abs(np.random.normal(10000, 5000)).astype(int)
    
    # 4. Generate "Ground Truth" Creditworthiness (The 'Real' Risk)
    # Financial Score: Higher Income & Employment + Lower Debt = Better Score
    financial_score = (income * 0.5) + (years_employed * 1000) - (debt * 1.2)
    
    # 5. Introduce Historical Bias in the TARGET variable (Training Data)
    # If bias_level is high, Group B needs a much higher score to be labeled "Good Credit"
    # This simulates biased historical data that the AI will train on.
    bias_penalty = np.where(groups == 'Group B', 20000 * bias_level, 0)
    
    final_score = financial_score - bias_penalty + np.random.normal(0, 2000, n_samples) # Add noise
    
    # Determine labels (1 = Repaid/Approve, 0 = Default/Reject)
    # Threshold set to approve roughly top 60%
    threshold = np.percentile(final_score, 40)
    labels = (final_score > threshold).astype(int)
    
    df = pd.DataFrame({
        "Applicant_ID": ids,
        "Group": groups,
        "Income": income,
        "Debt": debt,
        "Years_Employed": years_employed,
        "Financial_Score": financial_score, # The "Fair" metric
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
df = generate_data(n_samples=n_applicants, bias_level=bias_slider)

# Train Model on the Fly
X = df]
X = pd.get_dummies(X, columns=['Group'], drop_first=True) # Encode Group for AI
y = df['Historical_Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# The "Black Box" (Random Forest)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Add predictions to the dataframe for visualization
df = clf.predict(pd.get_dummies(df], columns=['Group'], drop_first=True))
df['AI_Label'] = df.apply(lambda x: "✅ Approved" if x == 1 else "❌ Rejected")


# --- TABS FOR STAGES ---
tab1, tab2, tab3, tab4 = st.tabs()

with tab1:
    st.subheader("Step 1: Understand the Training Data")
    st.markdown("This data represents historical loan outcomes. The AI learns from this history. If history is biased, the AI will be biased.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Applicants", n_applicants)
        st.metric("Group A Count", len(df[df['Group']=='Group A']))
    with col2:
        st.metric("Overall Default Rate", f"{(1 - df['Historical_Outcome'].mean())*100:.1f}%")
        st.metric("Group B Count", len(df[df['Group']=='Group B']))

    # Income Distribution Plot
    fig = px.histogram(df, x="Income", color="Group", barmode="overlay", title="Income Distribution by Group",
                       color_discrete_map={'Group A':'#636EFA', 'Group B':'#EF553B'})
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Raw Data Sample"):
        st.dataframe(df.head(10))

with tab2:
    st.subheader("Step 2: The Black Box Decision")
    st.markdown(f"""
    We trained a **Random Forest** algorithm on the data. 
    
    **Model Accuracy:** `{accuracy*100:.1f}%`
    
    The model is now making decisions on new applicants. Below, we see a scatter plot of **Income vs. Debt**. 
    The color indicates whether the AI Approved or Rejected them.
    """)
    
    # Interactive Scatter Plot
    fig_scatter = px.scatter(
        df, 
        x="Income", 
        y="Debt", 
        color="AI_Label",
        symbol="Group",
        hover_data=,
        color_discrete_map={"✅ Approved": "green", "❌ Rejected": "red"},
        title="AI Decision Boundary: Income vs Debt (Shape = Demographic Group)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.warning("""
    **Look closely:** Are there individuals from **Group B (Diamonds)** who have high income and low debt (top left area) but are still **Red (Rejected)**?
    This suggests the model has learned a 'proxy' for bias.
    """)

with tab3:
    st.subheader("Step 3: Fairness & Ethics Audit")
    
    # --- METRICS CALCULATION ---
    group_a = df[df['Group'] == 'Group A']
    group_b = df[df['Group'] == 'Group B']
    
    approve_a = group_a.mean()
    approve_b = group_b.mean()
    
    disparate_impact = approve_b / approve_a if approve_a > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Group A Approval Rate", f"{approve_a*100:.1f}%")
    col2.metric("Group B Approval Rate", f"{approve_b*100:.1f}%")
    col3.metric("Disparate Impact Ratio", f"{disparate_impact:.2f}", 
                delta="Fair > 0.8" if disparate_impact > 0.8 else "Bias Detected", delta_color="normal" if disparate_impact > 0.8 else "inverse")

    if disparate_impact < 0.8:
        st.error(f"⚠️ **Bias Detected:** Group B is only {disparate_impact*100:.0f}% as likely to be approved as Group A. This violates the '80% Rule' often used in fair lending audits.")
    else:
        st.success("✅ **Fairness Check Passed:** The model appears to treat groups relatively equally.")

    st.markdown("### Feature Importance (Global Explainability)")
    st.markdown("What features is the model *actually* using to make decisions?")
    
    # Feature Importance Plot
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': clf.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h', title="What drives the AI's decision?")
    st.plotly_chart(fig_imp, use_container_width=True)
    
    if 'Group_Group B' in importances['Feature'].values:
        imp_score = importances.loc[importances['Feature'] == 'Group_Group B', 'Importance'].values
        if imp_score > 0.05:
            st.error(f"🚨 **Ethical Alert:** The model is explicitly using 'Group_Group B' (Importance: {imp_score:.2f}) to make decisions. In many jurisdictions, this is illegal 'Redlining'.")

with tab4:
    st.subheader("Step 4: The Simulation Task")
    
    # Find a specific victim of the Black Box
    # Criteria: Good Financial Score, Group B, Rejected by AI
    victims = df[
        (df['Group'] == 'Group B') & 
        (df > df.median()) & 
        (df == 0)
    ].sort_values(by='Financial_Score', ascending=False)
    
    if not victims.empty:
        victim = victims.iloc
        st.markdown(f"""
        ### Case File: Applicant #{victim}
        **Name (Pseudonym):** Alex Chen  
        **Demographic:** Group B  
        **Income:** ${victim['Income']:,}  
        **Debt:** ${victim:,}  
        **Years Employed:** {victim} years  
        
        ---
        **AI Decision:** ❌ REJECTED  
        **Confidence:** 92%
        
        **Your Task:**
        Alex has written to the bank demanding an explanation. He knows his income is high and debt is low. 
        As the Bank Officer, draft a letter explaining *specifically* why the AI rejected him, based *only* on the Model Logic shown in Tab 2 & 3.
        """)
        
        explanation = st.text_area("Draft your explanation letter here:", height=200)
        
        if st.button("Submit Explanation"):
            st.info("Analyze your answer: Did you mention his 'Group'? Did you mention 'Income'? If the AI prioritized Group status over Income (as seen in Tab 3), can you legally tell him that?")
            
    else:
        st.success("In this specific simulation run, no obvious high-performing victims were found. Try increasing the Bias Slider in the sidebar to generate more unfair rejections.")

# Footer
st.markdown("---")
st.caption("Principles of Fintech | Module 4: AI Ethics & Explainability | Built with Streamlit")