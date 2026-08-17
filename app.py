import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import datetime
import json
import os
from google import genai
from google.genai import types

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Knowledge Retention Predictor",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------------
# Database Setup
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect("retention_data.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS study_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            study_date TEXT,
            initial_score REAL,
            days_since INTEGER,
            retained_percentage REAL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            quiz_date TEXT,
            score REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------
# Gemini API Initialization
# ---------------------------------------------------------
@st.cache_resource
def get_gemini_client():
    # Retrieve key from Streamlit secrets or local environment
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Gemini API Key missing! Set 'GEMINI_API_KEY' in Streamlit Secrets.")
        return None
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Failed to initialize Gemini Client: {e}")
        return None

client = get_gemini_client()

# ---------------------------------------------------------
# Mathematical Forgetting Curve Prediction
# ---------------------------------------------------------
def predict_retention(initial_score, days, decay_factor=0.08):
    # Ebbinghaus Forgetting Curve formula: R = e^(-t/S)
    retention = initial_score * np.exp(-decay_factor * days)
    return max(0.0, min(100.0, retention))

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.title(" Navigation")

# Explicitly defining app_mode to prevent UnboundLocalError
app_mode = st.sidebar.radio(
    "Select Mode",
    ["Dashboard & Predictor", "Take Quiz", "Session Logs"]
)

# ---------------------------------------------------------
# Mode 1: Dashboard & Predictor
# ---------------------------------------------------------
if app_mode == "Dashboard & Predictor":
    st.title("🧠 AI Knowledge Retention Predictor")
    st.write("Track and predict your memory retention using cognitive decay models.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Log New Study Topic")
        topic = st.text_input("Topic Name", placeholder="e.g., Computer Vision & YOLOv8")
        study_date = st.date_input("Study Date", datetime.date.today())
        initial_score = st.slider("Initial Mastery Score (%)", 0, 100, 85)
        days_ahead = st.slider("Predict Retention Days Ahead", 1, 60, 7)

        if st.button("Calculate & Save Session"):
            if topic:
                days_since = (datetime.date.today() - study_date).days
                predicted_ret = predict_retention(initial_score, days_ahead)
                
                conn = sqlite3.connect("retention_data.db")
                c = conn.cursor()
                c.execute(
                    "INSERT INTO study_sessions (topic, study_date, initial_score, days_since, retained_percentage) VALUES (?, ?, ?, ?, ?)",
                    (topic, str(study_date), initial_score, days_ahead, predicted_ret)
                )
                conn.commit()
                conn.close()
                
                st.success(f"Session saved! Predicted retention after {days_ahead} days: {predicted_ret:.1f}%")
            else:
                st.warning("Please enter a topic name.")

    with col2:
        st.subheader("Retention Projection Curve")
        days_range = np.arange(0, 31, 1)
        retention_curve = [predict_retention(initial_score, d) for d in days_range]
        
        df_chart = pd.DataFrame({
            "Days": days_range,
            "Predicted Retention (%)": retention_curve
        }).set_index("Days")
        
        st.line_chart(df_chart)

# ---------------------------------------------------------
# Mode 2: Take Quiz (Gemini API Integration)
# ---------------------------------------------------------
elif app_mode == "Take Quiz":
    st.title("🎯 AI-Generated Knowledge Quiz")
    st.write("Test your retention on demand using Gemini AI.")

    client = get_gemini_client()

    quiz_topic = st.text_input("Enter Topic for Quiz", placeholder="e.g., Machine Learning Fundamentals")
    
    if st.button("Generate Quiz") and quiz_topic:
        if client is None:
            st.error("Gemini API Client is not configured properly.")
        else:
            with st.spinner("Generating quiz questions..."):
                prompt = f"Create a 3-question multiple-choice quiz about '{quiz_topic}'. Return JSON array of objects, each with 'question', 'options' (array of 4 string choices), and 'answer' (exact matching option string)."
                try:
                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json"
                        )
                    )
                    quiz_data = json.loads(response.text)
                    st.session_state["active_quiz"] = quiz_data
                    st.session_state["quiz_topic"] = quiz_topic
                except Exception as e:
                    st.error(f"Error generating quiz from Gemini API: {e}")

    if "active_quiz" in st.session_state:
        st.subheader(f"Quiz: {st.session_state['quiz_topic']}")
        user_answers = []
        
        for idx, q in enumerate(st.session_state["active_quiz"]):
            st.write(f"**Q{idx+1}: {q['question']}**")
            user_ans = st.radio(f"Select option for Q{idx+1}", q["options"], key=f"q_{idx}")
            user_answers.append((user_ans, q["answer"]))
        
        if st.button("Submit Quiz"):
            score = sum([1 for user, correct in user_answers if user == correct]) / len(user_answers) * 100
            st.metric("Your Score", f"{score:.1f}%")
            
            conn = sqlite3.connect("retention_data.db")
            c = conn.cursor()
            c.execute(
                "INSERT INTO quiz_results (topic, quiz_date, score) VALUES (?, ?, ?)",
                (st.session_state["quiz_topic"], str(datetime.date.today()), score)
            )
            conn.commit()
            conn.close()
            st.success("Quiz result recorded successfully!")

# ---------------------------------------------------------
# Mode 3: Session Logs
# ---------------------------------------------------------
elif app_mode == "Session Logs":
    st.title("📋 Recorded Sessions & Quiz History")
    
    conn = sqlite3.connect("retention_data.db")
    df_sessions = pd.read_sql_query("SELECT * FROM study_sessions", conn)
    df_quizzes = pd.read_sql_query("SELECT * FROM quiz_results", conn)
    conn.close()

    st.subheader("Study Sessions Log")
    if not df_sessions.empty:
        st.dataframe(df_sessions, use_container_width=True)
    else:
        st.info("No study sessions recorded yet.")

    st.subheader("Quiz Results Log")
    if not df_quizzes.empty:
        st.dataframe(df_quizzes, use_container_width=True)
    else:
        st.info("No quiz results recorded yet.")