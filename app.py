# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import time

st.set_page_config(page_title="Black Box Credit Game", layout="centered")

# === CONFIGURATION (YOU EDIT THIS SECTION ONLY) ===
APPLICANTS = [
    {"id": 1, "name": "Ana Lopez",       "income": 65000, "debt": 8000,  "age": 28, "years_employed": 5},
    {"id": 2, "name": "Marcus Weber",    "income": 45000, "debt": 15000, "age": 42, "years_employed": 10},
    {"id": 3, "name": "Li Zhang",        "income": 85000, "debt": 5000,  "age": 35, "years_employed": 8},
    {"id": 4, "name": "Omar Hassan",     "income": 38000, "debt": 22000, "age": 31, "years_employed": 3},
    {"id": 5, "name": "Sofia Petrov",    "income": 52000, "debt": 12000, "age": 29, "years_employed": 6},
]

# Round 1 – Transparent scorecard (you can change these rules anytime)
def transparent_score(app):
    score = 0
    reason = []
    if app["income"] > 50000:    score += 10; reason.append("+10 Income > $50k")
    else:                        reason.append("±0 Income ≤ $50k")
    if app["debt"] < 10000:      score += 10; reason.append("+10 Debt < $10k")
    else:                        reason.append("±0 Debt ≥ $10k")
    if app["years_employed"] > 5:score += 5;  reason.append("+5 Employed >5y")
    return score, " | ".join(reason)

TRANSPARENT_THRESHOLD = 15

# Round 2 – Black-box (hidden logic + intentional surprise)
def blackbox_decision(app):
    # Hidden real model (you can make this as complex and biased as you want)
    raw_score = (
        0.4 * (app["income"]/1000) +
        -0.6 * (app["debt"]/1000) +
        0.3 * app["years_employed"] +
        np.random.normal(0, 8)            # noise
    )
    # INTENTIONAL SURPRISE: Ana Lopez (high transparent score) gets rejected because of age < 30 "proxy"
    if app["name"] == "Ana Lopez":
        confidence = 98
        decision = "Reject"
    else:
        confidence = 95 + abs(np.random.normal(0, 4))
        decision = "Approve" if raw_score > 18 else "Reject"
    return decision, round(confidence, 1)

# Google Sheets leaderboard (optional – comment out if you don't want it)
USE_GOOGLE_SHEETS = True
if USE_GOOGLE_SHEETS:
    try:
        scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        gc = gspread.authorize(creds)
        leaderboard = gc.open("Fintech_BlackBox_Leaderboard").sheet1
    except:
        USE_GOOGLE_SHEETS = False
        st.warning("Google Sheets leaderboard not connected")

# === SESSION STATE ===
if "team_name" not in st.session_state:
    st.session_state.team_name = ""
if "round" not in st.session_state:
    st.session_state.round = 0
if "decisions_r1" not in st.session_state:
    st.session_state.decisions_r1 = {}
if "decisions_r2" not in st.session_state:
    st.session_state.decisions_r2 = {}
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

# === TEAM LOGIN ===
if not st.session_state.team_name:
    st.title("🏦 Black Box Credit Simulation Game")
    st.write("### Principles of FinTech – Explainability Module")
    team = st.text_input("Enter your team name", max_chars=30)
    if st.button("Start Game"):
        if team.strip():
            st.session_state.team_name = team.strip()
            st.rerun()
        else:
            st.error("Please enter a team name")
    st.stop()

# === MAIN GAME ===
st.title(f"Team: **{st.session_state.team_name}**")

elapsed = int(time.time() - st.session_state.start_time)
mins = elapsed // 60
secs = elapsed % 60
st.sidebar.metric("Time elapsed", f"{mins:02d}:{secs:02d}")
if elapsed > 45*60:
    st.sidebar.error("⏰ Time's up!")

tab1, tab2, tab3, tab4 = st.tabs(["Instructions", "Round 1 – Transparent", "Round 2 – Black Box", "Leaderboard"])

with tab1:
    st.header("Activity Instructions")
    st.write("""
    1. **Round 1**: Use the transparent scorecard to decide on 5 loan applications.  
    2. **Round 2**: Use the bank's new AI "Black Box" system on the SAME applicants.  
    3. One applicant who was approved in Round 1 will now be rejected.  
    4. You will have to write an explanation letter to this customer → you will discover it is impossible.  
    5. Discussion: GDPR Art. 22, right to explanation, accuracy vs interpretability trade-off.
    """)

with tab2:
    st.header("Round 1 – Transparent Scorecard Model")
    st.write("**Rules**: Income > $50k → +10 | Debt < $10k → +10 | Employed >5y → +5 | **Threshold = 15**")

    df1 = []
    for app in APPLICANTS:
        score, reason = transparent_score(app)
        decision = "Approve" if score >= TRANSPARENT_THRESHOLD else "Reject"
        df1.append({"Name": app["name"], "Score": score, "Reason": reason, "Decision": decision})
    st.dataframe(pd.DataFrame(df1), use_container_width=True, hide_index=True)

    st.info("You can freely decide to follow or override the model in Round 1 (in practice banks did this). Record your final decisions below if you want them saved.")
    for app in APPLICANTS:
        col1, col2 = st.columns([3,1])
        with col1:
            st.write(f"**{app['name']}**")
        with col2:
            decision = st.selectbox("Decision", ["Approve", "Reject"], key=f"r1_{app['id']}")
            st.session_state.decisions_r1[app["name"]] = decision

with tab3:
    st.header("Round 2 – AI Black-Box Model (Production System)")
    st.write("The new system is much more accurate on historical data (backtested 94% vs your 81%), but the logic is proprietary.")

    surprise_name = "Ana Lopez"
    for app in APPLICANTS:
        decision, confidence = blackbox_decision(app)
        if app["name"] == surprise_name:
            st.error(f"🚨 {app['name']} → **{decision}** ({confidence}% confidence)")
        else:
            st.write(f"{app['name']} → **{decision}** ({confidence}% confidence)")

    st.session_state.decisions_r2 = {app["name"]: blackbox_decision(app)[0] for app in APPLICANTS}

    st.divider()
    st.subheader("📝 Task: Write the explanation letter to Ana Lopez")
    explanation = st.text_area(
        "Explain to Ana Lopez (in a professional tone) why her loan was rejected. You may ONLY use the information provided by the black-box system.",
        height=300,
        key="letter"
    )

    if st.button("Submit Letter & Finish"):
        correct_rejection = st.session_state.decisions_r2[surprise_name] == "Reject"
        score = 100 if explanation else 50  # you can make scoring more sophisticated
        st.session_state.final_score = score
        if USE_GOOGLE_SHEETS:
            leaderboard.append_row([datetime.now().strftime("%Y-%m-%d %H:%M"), st.session_state.team_name, score, explanation[:200]])
        st.success(f"Submitted! Your team score: {score}/100")

with tab4:
    st.header("🏆 Live Leaderboard")
    if USE_GOOGLE_SHEETS:
        data = leaderboard.get_all_records()
        if data:
            df_lb = pd.DataFrame(data)
            df_lb = df_lb.sort_values("Score", ascending=False)
            st.dataframe(df_lb[["Team", "Score", "Timestamp"]], use_container_width=True, hide_index=True)
        else:
            st.info("No submissions yet")
    else:
        st.info("Leaderboard disabled")

# Download letter
if "letter" in st.session_state:
    st.download_button(
        "Download Explanation Letter (for the lawsuit role-play)",
        st.session_state.letter,
        file_name=f"Explanation_{st.session_state.team_name}_AnaLopez.txt"
    )