import os
import sqlite3
import datetime
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="AI Knowledge Retention Predictor",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------------
# UPDATED & FIXED SDK INITIALIZATION
# ---------------------------------------------------------
client = None
selected_model_name = None
genai_available = False

try:
    from google import genai
    genai_available = True
except ImportError:
    genai_available = False

raw_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if genai_available and raw_key:
    try:
        clean_key = str(raw_key).strip().strip('"').strip("'")
        client = genai.Client(api_key=clean_key)
        
        # Explicit, well-formatted candidate list
        candidate_models = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-pro"]
        
        for m in candidate_models:
            try:
                # Sanitize string to prevent unexpected format errors
                model_id = m.replace("models/", "")
                client.models.generate_content(
                    model=model_id,
                    contents="ping"
                )
                selected_model_name = model_id
                break
            except Exception:
                continue
    except Exception:
        client = None
        selected_model_name = None

# Database Setup
def init_db():
    conn = sqlite3.connect("retention_predictor.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS study_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            study_date TEXT,
            retention_score REAL,
            quiz_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_log(topic, study_date, retention_score, quiz_score):
    conn = sqlite3.connect("retention_predictor.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO study_logs (topic, study_date, retention_score, quiz_score)
        VALUES (?, ?, ?, ?)
    """, (topic, str(study_date), retention_score, quiz_score))
    conn.commit()
    conn.close()

def get_logs():
    conn = sqlite3.connect("retention_predictor.db")
    c = conn.cursor()
    c.execute("SELECT topic, study_date, retention_score, quiz_score, timestamp FROM study_logs ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows

init_db()

def calculate_ebbinghaus_retention(days, strength=1.5):
    days_clean = np.maximum(days, 0)
    return np.exp(-days_clean / strength) * 100

st.title("🧠 AI Knowledge Retention Predictor")

with st.sidebar:
    st.header("⚙️ Configuration")
    if not genai_available:
        st.error("❌ `google-genai` missing in requirements.txt")
    elif not raw_key:
        st.warning("⚠️ API Key missing in Secrets")
    elif client and selected_model_name:
        st.success(f"✅ Connected (`{selected_model_name}`)")
    else:
        st.error("❌ Invalid API Key or Authentication Error")
        
    app_mode = st.radio("Select View", ["Dashboard & Predictor", "AI Quiz Generator", "Learning History Log"])

if app_mode == "Dashboard & Predictor":
    st.subheader("📊 Memory Decay Analysis")
    col1, col2 = st.columns([1, 1.5])
    with col1:
        topic_name = st.text_input("Topic Name", value="Computer Vision & Machine Learning")
        date_learned = st.date_input("Date Studied", value=datetime.date.today() - datetime.timedelta(days=4))
        memory_factor = st.slider("Memory Strength", 0.5, 5.0, 1.5, 0.1)
        days_passed = (datetime.date.today() - date_learned).days
        current_retention = float(calculate_ebbinghaus_retention(days_passed, memory_factor))
        st.metric("Retention Rate", f"{current_retention:.1f}%")
        
    with col2:
        x_days = np.linspace(0, 14, 100)
        y_retention = calculate_ebbinghaus_retention(x_days, memory_factor)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_days, y=y_retention, mode='lines', name='Decay Curve'))
        fig.add_trace(go.Scatter(x=[days_passed], y=[current_retention], mode='markers', name='Today', marker=dict(size=12, color='red')))
        fig.update_layout(xaxis_title="Days", yaxis_title="Retention (%)", height=350)
        st.plotly_chart(fig, use_container_width=True)

elif app_mode == "AI Quiz Generator":
    st.subheader("⚡ Automated AI Assessment")
    quiz_topic = st.text_input("Enter Topic", value="Data Structures & Algorithms")
    num_questions = st.slider("Number of Questions", 1, 5, 3)
    
    # Disable button completely if model is invalid or None
    is_disabled = (client is None or selected_model_name is None)
    
    if st.button("Generate Quiz with Gemini AI", disabled=is_disabled):
        with st.spinner("Generating quiz..."):
            try:
                prompt = f"Create {num_questions} multiple choice questions for '{quiz_topic}' with options and answers."
                
                # Format check: Pass clean string to model parameter
                response = client.models.generate_content(
                    model=str(selected_model_name).strip(),
                    contents=prompt
                )
                st.session_state["active_quiz"] = response.text
            except Exception as e:
                st.error(f"Failed to query Gemini API: {e}")

    if "active_quiz" in st.session_state:
        st.info(st.session_state["active_quiz"])
        user_score = st.number_input("Score (%)", 0.0, 100.0, 85.0)
        if st.button("Save Session"):
            save_log(quiz_topic, datetime.date.today(), 100.0, user_score)
            st.success("Log saved!")

elif app_mode == "Learning History Log":
    st.subheader("📜 Session History")
    logs = get_logs()
    if logs:
        st.dataframe([{"Topic": r[0], "Date": r[1], "Retention": f"{r[2]:.1f}%", "Score": f"{r[3]:.1f}%"} for r in logs], use_container_width=True)
    else:
        st.info("No recorded logs.")