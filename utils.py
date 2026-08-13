import json
import requests
import streamlit as st

API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
# Paste your API key here (Starts with AIzaSy...)

# Updated model endpoint to gemini-3.6-flash
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={API_KEY}"


def make_gemini_request(prompt: str):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }
    
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(GEMINI_URL, json=payload, headers=headers)
    res_data = response.json()

    if "error" in res_data:
        error_info = res_data["error"]
        code = error_info.get("code", "Unknown Code")
        message = error_info.get("message", "No message provided")
        raise Exception(f"Google API Error [{code}]: {message}")

    if "candidates" not in res_data:
        raise Exception(f"API returned unexpected response: {res_data}")

    text_content = res_data["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text_content)


def analyze_retention_risk(topic: str, last_revised_days: int, quiz_score: float, difficulty: str):
    prompt = f"""
    Analyze learning retention risk:
    - Topic: {topic}
    - Days Since Last Revision: {last_revised_days}
    - Quiz Score: {quiz_score}%
    - Difficulty: {difficulty}

    Return ONLY a raw valid JSON object with keys: "retention_score" (number), "forgetting_window_days" (integer), "risk_level" (string: Low/Medium/High), "recommendation" (string).
    Do not use markdown backticks or extra text.
    """
    return make_gemini_request(prompt)


def generate_topic_quiz(topic: str, difficulty: str, num_questions: int = 3):
    prompt = f"""
    Generate a {num_questions}-question multiple choice quiz on '{topic}' at '{difficulty}' level.

    Return ONLY a raw valid JSON object with key "quiz" containing an array of objects. Each object must have:
    "question" (string), "options" (array of 4 strings), "correct_answer" (string), "explanation" (string).
    Do not use markdown backticks or extra text.
    """
    return make_gemini_request(prompt)