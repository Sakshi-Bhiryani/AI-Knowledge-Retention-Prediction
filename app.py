import streamlit as st
import requests
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="AI Knowledge Retention Predictor",
    page_icon="🧠",
    layout="wide"
)

# Backend API Base URL
BACKEND_URL = "http://127.0.0.1:8000"

st.title("🧠 AI Knowledge Retention Predictor")
st.caption("Predictive learning insights and forgetting curve analytics powered by Gemini 2.0 Flash")

# Tabs for Dashboard and Quiz
tab1, tab2 = st.tabs(["📊 Retention Dashboard", "📝 AI Quiz Generator"])

# -----------------------------------------------------------------------------
# TAB 1: RETENTION DASHBOARD
# -----------------------------------------------------------------------------
with tab1:
    st.header("Input Study Data")
    
    col_input, col_display = st.columns([1, 1])
    
    with col_input:
        topic = st.text_input("Topic Name", value="Data Structures - Binary Trees")
        last_revised = st.number_input("Days Since Last Revision", min_value=0, max_value=365, value=7)
        quiz_score = st.slider("Last Quiz Score (%)", min_value=0, max_value=100, value=65)
        difficulty = st.selectbox("Perceived Difficulty", ["Easy", "Medium", "Hard"])
        
        predict_btn = st.button("Predict Retention Risk", type="primary")

    if predict_btn:
        with st.spinner("Analyzing forgetting curve and calculating retention risk..."):
            try:
                payload = {
                    "topic": topic,
                    "last_revised_days": last_revised,
                    "quiz_score": quiz_score,
                    "difficulty": difficulty
                }
                
                response = requests.post(f"{BACKEND_URL}/predict-retention", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    retention_score = data.get("retention_score", 0)
                    forgetting_window = data.get("forgetting_window_days", 0)
                    risk_level = data.get("risk_level", "Medium")
                    recommendation = data.get("recommendation", "No recommendation provided.")
                    
                    with col_display:
                        st.subheader("Analysis Results")
                        
                        # Gauge Chart using Plotly
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=retention_score,
                            title={'text': "Retention Score (%)"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#FF4B4B" if retention_score < 50 else "#FFAA00" if retention_score < 75 else "#00CC96"},
                                'steps': [
                                    {'range': [0, 50], 'color': "#FFE5E5"},
                                    {'range': [50, 75], 'color': "#FFF5E5"},
                                    {'range': [75, 100], 'color': "#E5F9F3"}
                                ]
                            }
                        ))
                        fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        m1, m2 = st.columns(2)
                        m1.metric("Risk Level", risk_level)
                        m2.metric("Revise Within", f"{forgetting_window} Days")
                        
                        st.info(f"**Recommendation:** {recommendation}")
                else:
                    st.error(f"Error generating analysis: {response.json().get('detail', response.text)}")
            
            except Exception as e:
                st.error(f"Could not connect to backend server: {str(e)}")

# -----------------------------------------------------------------------------
# TAB 2: AI QUIZ GENERATOR
# -----------------------------------------------------------------------------
with tab2:
    st.header("Generate Practice Quiz")
    
    q_col1, q_col2 = st.columns([1, 1])
    
    with q_col1:
        quiz_topic = st.text_input("Quiz Topic", value=topic)
        quiz_difficulty = st.selectbox("Quiz Difficulty", ["Easy", "Medium", "Hard"], index=1, key="quiz_diff")
        num_questions = st.slider("Number of Questions", min_value=1, max_value=5, value=3)
        
        generate_quiz_btn = st.button("Generate Quiz")

    if generate_quiz_btn:
        with st.spinner("Generating custom questions with Gemini..."):
            try:
                payload = {
                    "topic": quiz_topic,
                    "difficulty": quiz_difficulty,
                    "num_questions": num_questions
                }
                
                response = requests.post(f"{BACKEND_URL}/generate-quiz", json=payload)
                
                if response.status_code == 200:
                    res_data = response.json()
                    
                    # Safely handle string vs dict responses
                    if isinstance(res_data, dict) and "quiz" in res_data:
                        questions = res_data["quiz"]
                    elif isinstance(res_data, list):
                        questions = res_data
                    else:
                        questions = []
                    
                    st.session_state["current_quiz"] = questions
                else:
                    st.error(f"Error generating quiz: {response.json().get('detail', response.text)}")
            
            except Exception as e:
                st.error(f"Could not connect to backend server: {str(e)}")

    # Display Quiz Questions
    if "current_quiz" in st.session_state and st.session_state["current_quiz"]:
        st.subheader(f"Quiz on '{quiz_topic}'")
        questions = st.session_state["current_quiz"]
        
        with st.form("quiz_form"):
            user_answers = {}
            
            for idx, q in enumerate(questions):
                st.markdown(f"**Q{idx + 1}: {q.get('question', 'Question text missing')}**")
                options = q.get("options", [])
                user_answers[idx] = st.radio(f"Select option for Q{idx + 1}:", options, key=f"q_{idx}")
                st.divider()
            
            submit_quiz = st.form_submit_button("Submit Answers")
            
            if submit_quiz:
                score = 0
                total = len(questions)
                
                st.subheader("Quiz Results")
                for idx, q in enumerate(questions):
                    correct = q.get("correct_answer")
                    user_ans = user_answers.get(idx)
                    
                    if user_ans == correct:
                        score += 1
                        st.success(f"**Q{idx + 1}: Correct!** ({user_ans})")
                    else:
                        st.error(f"**Q{idx + 1}: Incorrect.** Your answer: {user_ans} | Correct: {correct}")
                    
                    st.caption(f"💡 **Explanation:** {q.get('explanation', 'None')}")
                
                final_percentage = round((score / total) * 100, 1)
                st.metric("Final Score", f"{score}/{total} ({final_percentage}%)")