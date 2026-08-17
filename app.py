import os
import sqlite3
import datetime
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Knowledge Retention Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. SDK INITIALIZATION & ERROR DEBUGGING
# ---------------------------------------------------------
client = None
selected_model_name = None
genai_available = False
auth_error_details = None

try:
    from google import genai
    genai_available = True
except ImportError as e:
    genai_available = False
    auth_error_details = f"ImportError: {e}"

# Read key from Secrets or Environment
raw_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if genai_available and raw_key:
    try:
        clean_key = str(raw_key).strip().strip('"').strip("'")
        client = genai.Client(api_key=clean_key)
        
        # Test models in order
        candidate_models = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-pro"]
        
        for m in candidate_models:
            try:
                model_id = m.replace("models/", "").strip()
                client.models.generate_content(
                    model=model_id,
                    contents="ping"
                )
                selected_model_name = model_id
                auth_error_details = None
                break
            except Exception as ping_err:
                auth_error_details = str(ping_err)
                continue
    except Exception as init_err:
        client = None
        selected_model_name = None
        auth_error_details = str(init_err)
# ---------------------------------------------------------
# 3. DATABASE SETUP (SQLite)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 4. EBBINGHAUS RETENTION ALGORITHM
# ---------------------------------------------------------
def calculate_ebbinghaus_retention(days, strength=1.5):
    """
    Calculates retention R = exp(-t / S) * 100
    Uses np.maximum to safely support float calculations and arrays.
    """
    days_clean = np.maximum(days, 0)
    return np.exp(-days_clean / strength) * 100

# ---------------------------------------------------------
# 5. SIDEBAR NAVIGATION & STATUS
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    
    if not genai_available:
        st.error("❌ `google-genai` package missing.")
    elif not raw_key:
        st.warning("⚠️ `GEMINI_API_KEY` missing in Streamlit Secrets.")
    elif client and selected_model_name:
        st.success(f"✅ Connected (`{selected_model_name}`)")
    else:
        st.error("❌ Invalid API Key or Authentication Error")
        if auth_error_details:
            st.caption(f"**Error Details:** `{auth_error_details}`")

# ---------------------------------------------------------
# 6. VIEW 1: DASHBOARD & PREDICTOR
# ---------------------------------------------------------
if app_mode == "Dashboard & Predictor":
    st.subheader("📊 Memory Decay Analysis")
    
    col_input, col_chart = st.columns([1, 1.5])
    
    with col_input:
        st.markdown("#### Topic Details")
        topic_name = st.text_input("Subject / Topic Name", value="Computer Vision & Machine Learning")
        date_learned = st.date_input("Date Studied", value=datetime.date.today() - datetime.timedelta(days=4))
        memory_factor = st.slider("Memory Strength (Days)", 0.5, 5.0, 1.5, 0.1, help="Higher values represent deeper initial understanding.")
        
        days_passed = (datetime.date.today() - date_learned).days
        current_retention = float(calculate_ebbinghaus_retention(days_passed, memory_factor))
        
        st.markdown("---")
        st.metric(label="Current Estimated Retention", value=f"{current_retention:.1f}%", delta=f"-{100 - current_retention:.1f}% decay")
        
        if current_retention < 60.0:
            st.error("⚠️ Retention threshold below 60%. Revision recommended today!")
        else:
            st.success("✅ Knowledge retention level is optimal.")
            
    with col_chart:
        st.markdown("#### Retentiveness Curve over 14 Days")
        x_days = np.linspace(0, 14, 100)
        y_retention = calculate_ebbinghaus_retention(x_days, memory_factor)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_days, y=y_retention, mode='lines', name='Decay Curve', line=dict(color='#0066CC', width=3)))
        fig.add_trace(go.Scatter(x=[days_passed], y=[current_retention], mode='markers', name='Today', marker=dict(size=14, color='red', symbol='cross')))
        fig.add_shape(type="line", x0=0, y0=60, x1=14, y1=60, line=dict(color="orange", dash="dash"))
        
        fig.update_layout(
            xaxis_title="Days Elapsed",
            yaxis_title="Retention Percentage (%)",
            yaxis=dict(range=[0, 105]),
            margin=dict(l=20, r=20, t=20, b=20),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 7. VIEW 2: AI QUIZ GENERATOR
# ---------------------------------------------------------
elif app_mode == "AI Quiz Generator":
    st.subheader("⚡ Automated AI Assessment")
    
    quiz_topic = st.text_input("Enter Topic to Test", value="Data Structures & Algorithms")
    num_questions = st.slider("Number of Questions", 1, 5, 3)
    
    # Disable button if connection is unverified or invalid
    is_disabled = (client is None or selected_model_name is None)
    
    if st.button("Generate Quiz with Gemini AI", disabled=is_disabled):
        with st.spinner("Generating quiz questions..."):
            try:
                prompt = (
                    f"Create {num_questions} multiple-choice quiz questions for the topic: '{quiz_topic}'. "
                    "For each question, provide 4 options (A, B, C, D) and clearly state the correct answer."
                )
                
                response = client.models.generate_content(
                    model=str(selected_model_name).strip(),
                    contents=prompt
                )
                st.session_state["active_quiz"] = response.text
            except Exception as e:
                st.error(f"Failed to query Gemini API: {e}")
                
    if "active_quiz" in st.session_state:
        st.markdown("### Generated Assessment")
        st.info(st.session_state["active_quiz"])
        
        st.markdown("---")
        user_score = st.number_input("Enter Your Performance Score (%)", min_value=0.0, max_value=100.0, value=85.0)
        
        if st.button("Save Assessment Log"):
            save_log(quiz_topic, datetime.date.today(), 100.0, user_score)
            st.success("Session saved to SQLite database!")

# ---------------------------------------------------------
# 8. VIEW 3: LEARNING HISTORY LOG
# ---------------------------------------------------------
elif app_mode == "Learning History Log":
    st.subheader("📜 Historical Session Logs")
    
    logs = get_logs()
    if logs:
        log_data = []
        for row in logs:
            log_data.append({
                "Topic": row[0],
                "Study Date": row[1],
                "Retention (%)": f"{row[2]:.1f}%",
                "Quiz Score (%)": f"{row[3]:.1f}%",
                "Logged At": row[4]
            })
        st.dataframe(log_data, use_container_width=True)
    else:
        st.info("No study logs recorded yet. Generate a quiz or run a prediction to save records.")
               