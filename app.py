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
def generate_data(n_samples=1000, bias_level=0.0, noise_level=0.2):
    """
    Enhanced data generator with configurable noise and realistic correlations
    """
    np.random.seed(42)
    
    # 1. Generate Demographics with controlled imbalance
    ids = range(1, n_samples + 1)
    groups = np.random.choice(['Group A', 'Group B'], size=n_samples, p=[0.7, 0.3])
    
    # 2. Generate Financials with systemic inequality
    income_base = np.where(groups == 'Group A', 65000, 55000)
    # Add realistic noise while preventing negative values
    income = np.maximum(20000, np.round(np.random.normal(income_base, 15000)).astype(int))
    
    # Years employed correlates with group (systemic barrier)
    years_employed_base = np.where(groups == 'Group A', 8, 5)
    years_employed = np.maximum(0, np.round(np.random.normal(years_employed_base, 4)).astype(int))
    
    # 3. Generate Debt with realistic constraints
    debt_base = np.where(groups == 'Group A', 8000, 12000)  # Systemic inequality
    debt = np.maximum(1000, np.round(np.abs(np.random.normal(debt_base, 6000))).astype(int))
    
    # 4. Financial Score (the "true" merit metric)
    financial_score = (income * 0.4) + (years_employed * 1500) - (debt * 1.0)
    financial_score = np.maximum(0, financial_score)  # Prevent negative scores
    
    # 5. Introduce Historical Bias with configurable noise
    bias_penalty = np.where(groups == 'Group B', 25000 * bias_level, 0)
    noise = np.random.normal(0, max(5000, 15000 * noise_level), n_samples)
    
    final_score = financial_score - bias_penalty + noise
    
    # Determine labels with adaptive threshold for ~60% approval rate
    threshold = np.percentile(final_score, 40)
    labels = (final_score > threshold).astype(int)
    
    df = pd.DataFrame({
        "Applicant_ID": ids,
        "Group": groups,
        "Income": income,
        "Debt": debt,
        "Years_Employed": years_employed,
        "Financial_Score": financial_score,
        "Historical_Outcome": labels,
        "Bias_Penalty": bias_penalty  # For audit purposes
    })
    
    return df

# ========================= MAIN APP =========================

st.title("🕵️‍♀️ Fintech Black Box Auditor")
st.markdown("""
**Workshop Module:** Principles of Fintech & Ethical AI  
**Objective:** Audit an AI Credit Scoring algorithm to detect bias and confront the explainability paradox.
""")

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    st.header("⚙️ Simulation Controls")
    st.info("""
    **Adjust these to explore different scenarios:**  
    - Higher *Bias Level* = More historical discrimination against Group B  
    - *Remove Group Data* = Attempt to build a "fair" AI (but bias may persist through proxies)  
    - *Noise Level* = Real-world data uncertainty
    """)
    
    n_applicants = st.slider("Applicant Pool Size", 500, 5000, 1000, 500)
    bias_slider = st.slider("Historical Bias Level", 0.0, 1.0, 0.7, 0.1,
                           help="0.0 = Perfectly fair historical decisions\n1.0 = Severe discrimination against Group B")
    noise_slider = st.slider("Data Noise Level", 0.0, 1.0, 0.3, 0.1,
                            help="Simulates real-world data imperfections")
    
    st.divider()
    st.subheader(" Mitigation Strategy")
    remove_group = st.checkbox(
        "✅ Remove demographic data before training",
        value=False,
        help="Simulates 'blind' AI development. Does this actually fix bias?"
    )
    
    st.divider()
    st.caption("""
    **Teaching Notes**  
    This simulation intentionally simplifies real-world complexity to reveal core ethical tensions in AI systems.
    """)

# Generate Data with noise parameter
df_original = generate_data(
    n_samples=n_applicants, 
    bias_level=bias_slider,
    noise_level=noise_slider
)

# --- MODEL TRAINING PIPELINE ---
def train_model(df, remove_group=False):
    """Encapsulated model training with feature engineering"""
    # Prepare features
    X = df.drop(columns=['Applicant_ID', 'Historical_Outcome', 'Bias_Penalty'])
    y = df['Historical_Outcome']
    
    # Remove protected attribute if requested
    if remove_group and 'Group' in X.columns:
        X = X.drop(columns=['Group'])
    
    # One-hot encode remaining categorical features
    X_encoded = pd.get_dummies(X, columns=['Group'], drop_first=True) if 'Group' in X.columns else X.copy()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    # Train model
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Apply to full dataset
    X_full = pd.get_dummies(
        df.drop(columns=['Applicant_ID', 'Historical_Outcome', 'Bias_Penalty']), 
        columns=['Group'], 
        drop_first=True
    ) if not remove_group and 'Group' in df.columns else df.drop(columns=['Applicant_ID', 'Historical_Outcome', 'Bias_Penalty', 'Group'], errors='ignore')
    
    # Align columns with training data
    for col in X_train.columns:
        if col not in X_full.columns:
            X_full[col] = 0
    X_full = X_full[X_train.columns]
    
    # Add predictions
    df_with_pred = df.copy()
    df_with_pred['AI_Prediction'] = clf.predict(X_full)
    df_with_pred['AI_Label'] = df_with_pred['AI_Prediction'].map({1: "✅ Approved", 0: "❌ Rejected"})
    
    return df_with_pred, clf, X_train, accuracy

# Train model with current settings
df_with_pred, clf, X_train, accuracy = train_model(df_original, remove_group=remove_group)

# --- ETHICAL METRICS CALCULATION ---
def calculate_fairness_metrics(df):
    """Calculate key fairness metrics with confidence intervals"""
    group_a = df[df['Group'] == 'Group A']
    group_b = df[df['Group'] == 'Group B']
    
    approve_a = group_a['AI_Prediction'].mean()
    approve_b = group_b['AI_Prediction'].mean()
    
    # Disparate impact with safety check
    disparate_impact = approve_b / approve_a if approve_a > 0.01 else 0
    
    # Statistical significance check (simplified)
    n_a = len(group_a)
    n_b = len(group_b)
    se = np.sqrt((approve_a * (1 - approve_a) / n_a) + (approve_b * (1 - approve_b) / n_b))
    ci_lower = disparate_impact - 1.96 * se
    ci_upper = disparate_impact + 1.96 * se
    
    return {
        'approve_a': approve_a,
        'approve_b': approve_b,
        'disparate_impact': disparate_impact,
        'ci_lower': max(0, ci_lower),
        'ci_upper': min(1, ci_upper),
        'stat_significant': (ci_lower > 0.8) or (ci_upper < 0.8)
    }

fairness_metrics = calculate_fairness_metrics(df_with_pred)

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Step 1: Historical Data", 
    "🤖 Step 2: AI Decisions", 
    "⚖️ Step 3: Fairness Audit", 
    "📝 Step 4: Explanation Challenge"
])

with tab1:
    st.subheader("Understanding Historical Bias")
    st.markdown("""
    This represents **historical loan decisions** that the AI learns from.  
    **Key insight:** AI systems replicate and amplify patterns in their training data.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Applicants", n_applicants)
        st.metric("Group B Proportion", f"{len(df_with_pred[df_with_pred['Group']=='Group B'])/n_applicants:.0%}")
    with col2:
        st.metric("Overall Approval Rate", f"{df_with_pred['Historical_Outcome'].mean()*100:.1f}%")
        group_b_approval = df_with_pred[df_with_pred['Group']=='Group B']['Historical_Outcome'].mean()
        st.metric("Group B Approval Rate", f"{group_b_approval*100:.1f}%")
    with col3:
        median_income_a = df_with_pred[df_with_pred['Group']=='Group A']['Income'].median()
        median_income_b = df_with_pred[df_with_pred['Group']=='Group B']['Income'].median()
        st.metric("Median Income (Group A)", f"${median_income_a:,.0f}")
        st.metric("Median Income (Group B)", f"${median_income_b:,.0f}")
    
    # Improved distribution visualization
    fig = px.histogram(
        df_with_pred, 
        x="Income", 
        color="Group", 
        barmode="overlay", 
        title="Income Distribution by Demographic Group",
        marginal="box",
        color_discrete_map={'Group A':'#1f77b4', 'Group B':'#d62728'},
        opacity=0.7,
        nbins=30
    )
    fig.update_layout(
        xaxis_title="Annual Income ($)",
        yaxis_title="Number of Applicants",
        legend_title="Demographic Group"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("🔍 Why this matters"):
        st.markdown("""
        **Systemic inequality** is baked into the data:
        - Group B has lower median income and higher debt
        - Historical approval rates differ significantly
        - The AI will learn these patterns unless explicitly mitigated
        
        > *"Garbage in, gospel out"* - AI systems treat biased historical data as ground truth
        """)

with tab2:
    st.subheader("The Black Box in Action")
    st.markdown(f"""
    We trained a **Random Forest classifier** on the historical data.  
    **Model Performance:** `{accuracy*100:.1f}%` accuracy on test data  
    **Critical Question:** *Is accuracy enough when fairness is at stake?*
    """)
    
    # Enhanced scatter plot with decision boundaries
    fig_scatter = px.scatter(
        df_with_pred, 
        x="Income", 
        y="Debt", 
        color="AI_Label",
        symbol="Group",
        size="Financial_Score",
        size_max=15,
        hover_data={
            "Applicant_ID": True,
            "Years_Employed": True,
            "Financial_Score": ":,.0f",
            "AI_Label": False,
            "Group": True
        },
        color_discrete_map={"✅ Approved": "#2ca02c", "❌ Rejected": "#d62728"},
        title="AI Decisions: Income vs. Debt (Bubble size = Financial Score)",
        labels={
            "Income": "Annual Income ($)",
            "Debt": "Outstanding Debt ($)",
            "Financial_Score": "Financial Health Score"
        }
    )
    
    # Add quadrant lines for better interpretation
    fig_scatter.add_hline(y=df_with_pred['Debt'].median(), line_dash="dash", line_color="gray")
    fig_scatter.add_vline(x=df_with_pred['Income'].median(), line_dash="dash", line_color="gray")
    
    fig_scatter.update_layout(
        legend_title="AI Decision & Group",
        xaxis_title="Annual Income ($)",
        yaxis_title="Outstanding Debt ($)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.warning("""
    **🔍 Investigate these patterns:**
    - Are there **high-income, low-debt applicants from Group B (♦)** who were **rejected (red)**?
    - How does bubble size (financial score) correlate with decisions?
    - If you enabled *"Remove demographic data"*, is bias still visible? (Hint: look for proxy variables)
    """)

with tab3:
    st.subheader("Fairness Audit Report")
    st.markdown("""
    **Regulatory Standard (80% Rule):**  
    For fair lending, the approval rate for disadvantaged groups should be at least 80% of the majority group's rate.
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Group A Approval Rate", 
        f"{fairness_metrics['approve_a']*100:.1f}%",
        delta=None
    )
    col2.metric(
        "Group B Approval Rate", 
        f"{fairness_metrics['approve_b']*100:.1f}%",
        delta=f"{(fairness_metrics['approve_b'] - fairness_metrics['approve_a'])*100:.1f} pp",
        delta_color="inverse"
    )
    col3.metric(
        "Disparate Impact Ratio", 
        f"{fairness_metrics['disparate_impact']:.2f}",
        delta="Acceptable" if fairness_metrics['disparate_impact'] >= 0.8 else "Violation",
        delta_color="normal" if fairness_metrics['disparate_impact'] >= 0.8 else "inverse"
    )

    # Visualize fairness metrics
    fairness_df = pd.DataFrame({
        'Group': ['Group A', 'Group B'],
        'Approval Rate': [fairness_metrics['approve_a'], fairness_metrics['approve_b']]
    })
    
    fig_fair = px.bar(
        fairness_df,
        x='Group',
        y='Approval Rate',
        color='Group',
        text='Approval Rate',
        title='Approval Rates by Demographic Group',
        range_y=[0, 1],
        color_discrete_map={'Group A': '#1f77b4', 'Group B': '#d62728'}
    )
    fig_fair.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig_fair.add_hline(
        y=fairness_metrics['approve_a'] * 0.8, 
        line_dash="dash", 
        line_color="red",
        annotation_text="80% Rule Threshold",
        annotation_position="top right"
    )
    st.plotly_chart(fig_fair, use_container_width=True)

    # Feature importance with enhanced interpretation
    st.markdown("### 🔍 What's Driving AI Decisions?")
    
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': clf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # Add interpretation hints
    importances['Interpretation'] = importances['Feature'].apply(
        lambda x: "⚠️ Protected attribute" if "Group" in x else 
                  "💰 Strong financial indicator" if x in ["Income", "Debt", "Years_Employed"] else
                  "❓ Potential proxy variable"
    )
    
    fig_imp = px.bar(
        importances, 
        x='Importance', 
        y='Feature', 
        color='Interpretation',
        orientation='h',
        title="Feature Importance Analysis",
        color_discrete_map={
            "⚠️ Protected attribute": "#ff7f0e",
            "💰 Strong financial indicator": "#2ca02c",
            "❓ Potential proxy variable": "#1f77b4"
        }
    )
    fig_imp.update_layout(
        yaxis={'categoryorder':'total ascending'},
        xaxis_title="Relative Importance",
        legend_title="Feature Type"
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # Ethical assessment
    st.markdown("### 🚨 Ethical Assessment")
    ethical_alert = False
    
    if not remove_group and 'Group_Group B' in importances['Feature'].values:
        imp_score = importances.loc[importances['Feature'] == 'Group_Group B', 'Importance'].values[0]
        if imp_score > 0.05:
            st.error(f"""
            ❗ **Critical Finding:** The model explicitly uses demographic group (**Group_Group B**) 
            with importance score {imp_score:.2f}. This likely violates anti-discrimination laws 
            (ECOA, GDPR) in most jurisdictions.
            """)
            ethical_alert = True
    
    if fairness_metrics['disparate_impact'] < 0.8:
        st.error(f"""
        ⚠️ **Fairness Violation Detected:**  
        Group B applicants are **{(fairness_metrics['disparate_impact']*100):.0f}%** as likely to be approved as Group A applicants.  
        This violates the **80% Rule** used by regulators worldwide for fair lending practices.
        """)
        ethical_alert = True
    
    if not ethical_alert:
        st.success("""
        ✅ **No Critical Violations Detected**  
        *Note: Absence of evidence isn't evidence of absence. Always conduct deeper audits with real-world data.*
        """)
    
    with st.expander("📘 Fairness Metrics Explained"):
        st.markdown("""
        **Disparate Impact Ratio** = (Approval Rate for Group B) / (Approval Rate for Group A)  
        - **≥ 0.8**: Generally acceptable under U.S. EEOC guidelines and EU AI Act  
        - **< 0.8**: Indicates potential illegal discrimination  
        
        **Why accuracy isn't enough:**  
        A model can be 90% accurate while systematically rejecting qualified applicants from protected groups.  
        *Example: In a population that's 90% Group A, a model that always approves Group A and always rejects Group B would be 90% accurate but completely unfair.*
        """)

with tab4:
    st.subheader("The Explanation Paradox")
    st.markdown("""
    **Scenario:** Alex Chen (Group B) was rejected despite strong finances.  
    **Your Task:** Draft an explanation letter that is both *truthful* and *legally compliant*.  
    
    > 📜 **Legal Constraint:** You **cannot mention demographic factors** (race, gender, etc.) in credit denial explanations under the Equal Credit Opportunity Act (ECOA).
    """)
    
    # Find high-performing rejected applicants from Group B
    median_score = df_with_pred['Financial_Score'].median()
    qualified_rejected = df_with_pred[
        (df_with_pred['Group'] == 'Group B') & 
        (df_with_pred['Financial_Score'] > median_score) & 
        (df_with_pred['AI_Prediction'] == 0)
    ].sort_values(by='Financial_Score', ascending=False)
    
    if not qualified_rejected.empty:
        # Select a representative case (not always the same)
        np.random.seed(42 + int(bias_slider * 100) + n_applicants)
        victim = qualified_rejected.sample(1).iloc[0]
        
        st.markdown(f"""
        ### 🧾 Applicant Profile: #{victim['Applicant_ID']}
        | **Attribute** | **Value** | **Context** |
        |---------------|-----------|-------------|
        | **Demographic** | Group B | *Protected characteristic - cannot be mentioned in explanation* |
        | **Income** | ${victim['Income']:,.0f} | Top {int(100 - (victim['Income'] / df_with_pred['Income'].max()) * 100)}% of applicants |
        | **Debt** | ${victim['Debt']:,.0f} | Below median debt level |
        | **Employment** | {victim['Years_Employed']} years | Stable employment history |
        | **Financial Score** | {victim['Financial_Score']:,.0f} | Top {int(100 - (victim['Financial_Score'] / df_with_pred['Financial_Score'].max()) * 100)}% of all applicants |
        | **AI Decision** | ❌ REJECTED | Confidence: {np.max(clf.predict_proba(X_train.loc[[victim.name]])):.0%} |
        """)
        
        st.divider()
        st.markdown("### 📝 Your Explanation Draft")
        
        # Provide template with ethical guardrails
        default_text = (
            "Dear Mr. Chen,\n\n"
            "Thank you for your loan application. After careful review using our automated decision system, "
            "we regret to inform you that your application was not approved at this time.\n\n"
            "Key factors that influenced this decision include:\n"
            "- [SPECIFIC FINANCIAL FACTORS FROM TABS 2-3]\n"
            "- [MENTION ACTUAL MODEL DRIVERS FROM FEATURE IMPORTANCE]\n\n"
            "Important: Per federal regulations, this decision was not based on your race, color, religion, "
            "national origin, sex, marital status, age, or receipt of public assistance.\n\n"
            "You have the right to request a free copy of the report we used and to dispute any inaccuracies.\n\n"
            "Sincerely,\n"
            "Automated Lending System"
        )
        
        explanation = st.text_area(
            "Draft your compliance-ready explanation:",
            value=default_text,
            height=300,
            help="Remember: You cannot mention demographic factors. Be specific about financial factors."
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("✅ Submit Explanation"):
                with col2:
                    st.info("""
                    **Reflection Questions:**  
                    - Did you mention any demographic factors? (Illegal)  
                    - Did you reference factors the model actually uses? (Check Tab 3)  
                    - If 'Group' was a top feature, how did you explain without mentioning it?  
                    - Is your explanation both truthful and compliant?  
                    
                    > This tension between **transparency** and **compliance** is the core challenge of ethical AI in finance.
                    """)
        
        st.divider()
        with st.expander("💡 Teaching Guidance: The Explanation Paradox"):
            st.markdown("""
            **This is the heart of the ethical dilemma:**  
            - If the model used 'Group' as a key feature (see Tab 3), you cannot truthfully explain the decision without mentioning it  
            - If you only mention financial factors, you're being dishonest about the real reason  
            - If you say "the algorithm decided", you're hiding behind the black box  
            
            **Real-world implications:**  
            - In 2019, Apple Card faced investigations for gender bias where qualified women received lower credit limits  
            - Under GDPR and CCPA, consumers have the "right to explanation" for automated decisions  
            - The EU AI Act requires high-risk systems (like credit scoring) to provide meaningful explanations  
            
            **Discussion prompts:**  
            1. Is it ever acceptable to use protected attributes if they improve accuracy?  
            2. Should banks be allowed to use AI for credit decisions if they can't explain them?  
            3. What technical solutions could resolve this paradox? (e.g., adversarial de-biasing, interpretable models)  
            """)
    else:
        st.success("""
        ✅ **No Clear Bias Cases Found**  
        With current settings, the model isn't flagrantly rejecting qualified Group B applicants.  
        **Try increasing the Bias Slider** to see how discrimination emerges, or **disable "Remove demographic data"** to see explicit bias.
        """)
        
        # Show counterfactual suggestion
        st.markdown("""
        ### 🔍 To see bias in action:
        1. Increase **Historical Bias Level** to 0.8+ 
        2. **Uncheck** "Remove demographic data before training"
        3. Look for Group B applicants in the top-left quadrant of Tab 2's chart (high income, low debt) who are rejected
        """)

# Footer with educational resources
st.markdown("---")
footer_col1, footer_col2 = st.columns([3, 1])
with footer_col1:
    st.caption("""
    **Principles of Fintech | Module 4: AI Ethics & Explainability**  
    Teaching Resources: [EU AI Act Guidelines](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) | 
    [CFPB Fair Lending Guidelines](https://www.consumerfinance.gov/compliance/compliance-resources/fair-lending-compliance/) |
    [Google's Responsible AI Practices](https://ai.google/responsibilities/)
    """)
with footer_col2:
    st.caption("Built with Streamlit")