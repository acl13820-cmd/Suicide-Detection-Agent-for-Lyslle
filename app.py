# ===========================================
# 🧠 Crisis Safety Agent Demo (Hybrid Version + Adjustable Threshold)
# ===========================================

import streamlit as st
import json
import re
import joblib
from openai import OpenAI

# This API key is for local use only. Please do not share this API key.
client = OpenAI(api_key="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")

# =====================
#  Load the ML model
# =====================
@st.cache_resource
def load_ml_model():
    vectorizer = joblib.load("myvec")
    model = joblib.load("lr_best")
    return vectorizer, model

vectorizer, ml_model = load_ml_model()

# =====================
# Step 1: Remove sensitive information from the text
# =====================
def clean_text(text):
    text = re.sub(r'\b(?!St|Ave|Rd|Blvd|Lane|Street|Road|Avenue|Terrace|Drive|Court)[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)
    text = re.sub(r'\d+\s[A-Z][a-z]+\s(St|Ave|Rd|Blvd|Lane|Street|Road|Avenue|Terrace|Drive|Court)\b', '[ADDRESS]', text)
    text = re.sub(r'\+?\d{1,3}[-\s(]?\d{2,4}[-\s)]?\d{3,4}[-\s]?\d{3,4}', '[PHONE]', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', '[DATE]', text)
    text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{4}\b', '[DATE]', text)
    text = re.sub(r'https?://\S+|t\.me/\S+|www\.\S+', '[URL]', text)
    text = re.sub(r'@\w+', '[HANDLE]', text)
    text = re.sub(r'\b[A-Z0-9]{8,}\b', '[ID]', text)
    return text

# =====================
# Step 2: GPT analysis
# =====================
def crisis_detector(text):
    prompt = f"""
    You are a safety monitoring agent.
    Determine if the message suggests self-harm, suicide, violence, or emotional distress.
    Return only valid JSON:
    {{"risk_level":"High/Medium/Low","category":"self-harm/harm-to-others/distress/none"}}

    Text: {text}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# =====================
# Step 3: ML predict
# =====================
def ml_predict(text, threshold=0.5):
    X = vectorizer.transform([text])
    prob = ml_model.predict_proba(X)[0]
    risk = "High" if prob[1] > threshold else "Low"
    category = "self-harm" if risk == "High" else "none"
    return {"risk_level": risk, "category": category, "probability": round(prob[1], 3)}

# =====================
# Step 4: result judgment
# =====================
def escalation_action(risk_level):
    if risk_level == "High":
        return "🚨 Alert therapist"
    elif risk_level == "Medium":
        return "🟡 Monitor and log"
    else:
        return "🟢 Safe – no action"

# =====================
# Step 5: Streamlit
# =====================
st.title("🧠 Crisis Safety Agent Demo (Hybrid Version)")
st.write("This version integrates GPT-based and ML-based risk detection modules with adjustable ML sensitivity.")

# sidebar slider
st.sidebar.header("⚙️ ML Sensitivity Control")
threshold = st.sidebar.slider("Risk Threshold (higher = stricter)", 0.0, 1.0, 0.5, 0.05)

text = st.text_area("Enter text to analyze:", height=150)

if st.button("Analyze"):
    cleaned = clean_text(text)

    
    crisis_json = crisis_detector(cleaned)
    try:
        gpt_result = json.loads(crisis_json)
        gpt_risk = gpt_result["risk_level"]
        gpt_category = gpt_result["category"]
    except:
        gpt_result = {"risk_level": "Low", "category": "none"}
        gpt_risk, gpt_category = "Low", "none"

    # ML output
    ml_result = ml_predict(cleaned, threshold)

    # decision
    risk_order = {"Low": 1, "Medium": 2, "High": 3}
    final_risk = gpt_risk if risk_order[gpt_risk] >= risk_order[ml_result["risk_level"]] else ml_result["risk_level"]
    final_category = gpt_category if final_risk == gpt_risk else ml_result["category"]

    # action
    action = escalation_action(final_risk)

    # show result
    st.subheader("📝 Original Input")
    st.write(text)

    st.subheader("🔒 Cleaned (De-identified) Text")
    st.write(cleaned)

    st.subheader("🔍 Machine Learning Result")
    st.json(ml_result)

    st.subheader("🧠 GPT Analysis Result")
    st.json(gpt_result)

    st.subheader("✅ Final Decision (Integrated)")
    st.markdown(f"### {final_risk} ({final_category})")

    st.subheader("⚙️ System Action")
    st.markdown(f"### {action}")
