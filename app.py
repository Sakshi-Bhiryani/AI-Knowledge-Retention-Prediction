import os
import streamlit as st
import google.generativeai as genai

# 1. Clear conflicting Google Cloud environment variables that force OAuth 2
for env_var in ["GOOGLE_APPLICATION_CREDENTIALS", "GCP_PROJECT", "GOOGLE_CLOUD_PROJECT"]:
    if env_var in os.environ:
        del os.environ[env_var]

# 2. Fetch and sanitize the API Key from Streamlit Secrets
raw_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

gemini_model = None

if raw_key:
    # Clean whitespace and accidental surrounding quotes
    clean_key = str(raw_key).strip().strip('"').strip("'")
    
    try:
        # Configure the global API key
        genai.configure(api_key=clean_key)
        
        # Instantiate model directly passing the api_key parameter
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        st.error(f"Initialization error: {e}")
else:
    st.warning("⚠️ GEMINI_API_KEY missing from Streamlit Secrets.")

# 3. Quiz Generation Trigger
if st.button("Generate Quiz with Gemini AI"):
    if gemini_model:
        try:
            response = gemini_model.generate_content("Generate 3 multiple choice questions on Data Structures.")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"Failed to query Gemini API: {e}")
    else:
        st.error("Gemini Model is not initialized. Please verify your GEMINI_API_KEY.")