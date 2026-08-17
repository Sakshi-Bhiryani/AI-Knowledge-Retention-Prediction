import os
import sqlite3
import datetime
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import google.generativeai as genai

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION (Must be the very first Streamlit command)
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Knowledge Retention Predictor",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------------
# 2. SAFE API KEY INITIALIZATION & FALLBACK
# ---------------------------------------------------------
api_key = None

# Check Streamlit Cloud Secrets safely
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        gemini_model = None
        st.error(f"Error configuring Gemini API: {e}")
else:
    gemini_model = None

# ---------------------------------------------------------
# 3. DATABASE SETUP (SQLite Persistence)
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect("user_history.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            study_date TEXT,
            last_score REAL,
            retention_percent REAL
        )
    """)
    conn.commit()
    conn.close()

def save_history(topic, study_date, score, retention):
    conn = sqlite3.connect("user_history.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO history (topic, study_date, last_score, retention_percent)
        VALUES (?, ?, ?, ?)
    """, (topic, str(study_date), score, retention))
    conn.commit()
    conn.close()

def fetch_history():
    conn = sqlite3.connect("user_history.db")
    c = conn.cursor()
    c.execute("SELECT topic, study_date, last_score, retention_percent FROM history ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows

init_db()

# ---------------------------------------------------------
# 4. MATHEMATICAL RETENTION LOGIC (Ebbinghaus Forgetting Curve)
# ---------------------------------------------------------
def calculate_retention(days_passed, S=1.5):
    """
    R = exp(-t / S)
    t = days passed
    S = memory strength factor (default 1.5 days for baseline recall)
    """
    if days_passed < 0:
        days_passed = 0
    return np.exp(-days_passed / S) * 100

# ---------------------------------------------------------
# 5. USER INTERFACE
# ---------------------------------------------------------
st.title("🧠 AI Knowledge Retention Predictor")
st.markdown("Predict memory decay using **Ebbinghaus models** and refresh knowledge via **Gemini AI quizzes**.")

if not api_key:
    st.warning("⚠️ `GEMINI_API_KEY` is not set. Go to Streamlit Cloud > Settings > Secrets and add `GEMINI_API_KEY = \"your_key\"` to enable AI quizzes.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Topic Setup")
    topic = st.text_input("Study Topic", value="Data Structures & Algorithms")
    study_date = st.date_input("Date Studied", value=datetime.date.today() - datetime.timedelta(days=3))
    memory_strength = st.slider("Initial Memory Strength (Days)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)

    days_elapsed = (datetime.date.today() - study_date).days
    current_retention = calculate_retention(days_elapsed, S=memory_strength)

    st.metric("Estimated Current Retention", f"{current_retention:.1f}%")

    if current_retention < 60.0:
        st.error("🔻 Memory decay threshold passed (< 60%). Revision recommended!")
    else:
        st.success("✅ Memory retention is optimal.")

with col2:
    st.subheader("2. Memory Decay Curve")
    days_range = np.linspace(0, 14, 100)
    retention_range = calculate_retention(days_range, S=memory_strength)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days_range, y=retention_range, mode='lines', name='Retention Curve', line=dict(color='#3366CC', width=3)))
    fig.add_trace(go.Scatter(x=[days_elapsed], y=[current_retention], mode='markers', name='Today', marker=dict(size=12, color='red')))
    fig.add_shape(type="line", x0=0, y0=60, x1=14, y1=60, line=dict(color="orange", dash="dash"))
    fig.update_layout(xaxis_title="Days Since Learning", yaxis_title="Retention (%)", yaxis=dict(range=[0, 105]), height=320)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# 6. DYNAMIC GEMINI AI QUIZ ENGINE
# ---------------------------------------------------------
st.subheader("3. Interactive AI Quiz Refresh")

if st.button("⚡ Generate AI Quiz", disabled=(gemini_model is None)):
    with st.spinner(f"Generating question for '{topic}' using Gemini..."):
        try:
            prompt = f"Create a single multiple choice question with 4 options (A, B, C, D) and indicate the correct answer for the topic: {topic}."
            response = gemini_model.generate_content(prompt)
            st.session_state["quiz_content"] = response.text
        except Exception as e:
            st.error(f"Failed to generate quiz: {e}")

if "quiz_content" in st.session_state:
    st.info(st.session_state["quiz_content"])
    score_input = st.slider("Record Your Performance Score (%)", 0, 100, 80)
    if st.button("Save Quiz Performance"):
        save_history(topic, study_date, score_input, current_retention)
        st.success("Saved session to database!")

# ---------------------------------------------------------
# 7. HISTORICAL LOGS
# ---------------------------------------------------------
st.divider()
st.subheader("4. Learning History")
logs = fetch_history()
if logs:
    st.table([{"Topic": r[0], "Study Date": r[1], "Last Score": f"{r[2]}%", "Retention": f"{r[3]:.1f}%"} for r in logs])
else:
    st.write("No session records found.")