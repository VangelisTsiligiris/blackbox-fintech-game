# app.py   ←  Fully working local + Streamlit Cloud version (no gspread!)
import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="Black Box Credit Game", layout="centered")

# ========================= CONFIGURATION (EDIT HERE) =========================
APPLICANTS = [
    {"id": 1, "name": "Ana López",       "income": 65000, "debt": 8000,   "age": 28, "years_employed": 5},
    {"id": 2, "name": "Marcus Weber",    "income": 45000, "debt": 15000,  "age": 42, "years_employed": 10},
    {"id": 3, "name": "Li Zhang",        "income": 85000, "debt": 5000,   "age": 35, "years_employed": 8},
    {"id": 4, "name": "Omar Hassan",     "income": 38000, "debt": 22000,  "age": 31, "years_employed": 3},
    {"id": 5, "name": "Sofia Petrova",   "income": 52000, "debt": 12000,  "age": 29, "years_employed": 6},
]

# Round 1 – Transparent model
def transparent_score(app):
    score = 0
    reasons = []
    if app["income"] > 50000:    score += 10; reasons.append("+10 Income > $50k")
    if app["debt"] < 10000:      score += 10; reasons.append("+10 Debt < $10k")
    if app["years_employed"] > 5:score += 5;  reasons.append("+5 Employed >5y")
    return score, " | ".join(reasons or ["No points"])

TRANSPARENT_THRESHOLD = 15

# Round 2 – Black-box with intentional surprise
def blackbox_decision(app):
    # Hidden logic – feel free to make it more complex
    base = (app["income"]/1000)*0.35 - (app["debt"]/1000)*0.65 + app["years_employed"]*0.4
    if app["age"] < 30:  # proxy for discrimination
        base -= 12
    decision = "Reject" if base < 18 else "Approve"
    confidence = round(90 + abs(np.random.normal(0, 7)), 1)
    
    # FORCE the twist: Ana López gets rejected even though transparent model approved her
    if app["name"] == "Ana López":
        return "Reject", 98.4
    return decision, confidence
# ===========================================================================

# Session state initialisation
if "team_name" not in st.session_state:
    st.session_state.team_name = ""
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "letter" not in st.session_state:
    st.session_state.letter = ""

# -------------------------- TEAM LOGIN --------------------------
if not st.session_state.team_name:
    st.title("🏦 Black Box Credit Simulation Game")
    st.markdown("### Principles of FinTech – Explainable AI Module")
    team = st.text_input("Enter your team name / group number", max_chars=30)
    if st.button("🚀 Start Game"):
        if team.strip():
            st.session_state.team_name = team.strip().upper()
            st.rerun()
        else:
            st.error("Please enter a team name")
    st.stop()

# -------------------------- MAIN APP --------------------------
st.title(f"Team: **{st.session_state.team_name}**")

# Timer
elapsed = int(time.time() - st.session_state.start_time)
mins, secs = divmod(elapsed, 60)
st.sidebar.metric("⏱ Time elapsed", f"{mins:02d}:{secs:02d}")
if elapsed > 45*60:
    st.sidebar.error("⏰ TIME'S UP! Submit your letter now.")

tab1, tab2, tab3, tab4 = st.tabs(["📋 Instructions", "1️⃣ Round 1 – Transparent", "2️⃣ Round 2 – Black Box", "✉️ Explanation Letter"])

with tab1:
    st.markdown("""
    ### How the activity works (45 minutes)
    1. **Round 1** → use the fully transparent scorecard  
    2. **Round 2** → use the new AI black-box (94% accuracy but proprietary)  
    3. One applicant who was approved in Round 1 is now rejected  
    4. You must write an explanation letter to that customer → you will realise it's impossible  
    5. Discussion on GDPR Article 22, right to explanation, accuracy vs interpretability trade-off
    """)

with tab2:
    st.header("Round 1 – Transparent Scorecard")
    st.write("**Everyone can see and understand the rules**")
    
    data = []
    for app in APPLICANTS:
        score, reason = transparent_score(app)
        decision = "✅ Approve" if score >= TRANSPARENT_THRESHOLD else "❌ Reject"
        data.append({"Name": app["name"], "Score": score, "Reasons": reason, "Model Decision": decision})
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

with tab3:
    st.header("Round 2 – New AI Black-Box Model")
    st.warning("🔒 This model has 94% accuracy on historical data (vs 81% for the old one) – but the logic is proprietary and protected.")
    
    for app in APPLICANTS:
        dec, conf = blackbox_decision(app)
        if app["name"] == "Ana López":
            st.error(f"🚨 **{app['name']}** → **{dec}** ({conf}% confidence)")
        else:
            st.success(f"**{app['name']}** → {dec} ({conf}% confidence)")

with tab4:
    st.header("✉️ Task: Write the rejection explanation letter")
    st.error("Ana López has sued the bank claiming discrimination. Draft a professional letter explaining why her application was rejected – using **only** the information the black-box provided.")
    
    st.session_state.letter = st.text_area(
        "Write your letter here (professional tone, address it to Ms. Ana López)",
        value=st.session_state.letter,
        height=450
    )
    
    st.download_button(
        label="⬇️ Download letter as .txt",
        data=st.session_state.letter,
        file_name=f"Rejection_Letter_Ana_Lopez_Team_{st.session_state.team_name}.txt",
        mime="text/plain"
    )

st.sidebar.success("When all teams are ready → collect the letters and discuss why proper explanation is impossible!")