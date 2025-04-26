import streamlit as st
from joblib import load
import re
import contractions
from num2words import num2words
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from scipy.sparse import hstack, csr_matrix
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os

# Set page config as the very first Streamlit command
st.set_page_config(page_title="Sentiment Classifier", layout="centered")

import nltk

def ensure_nltk_data():
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource)

ensure_nltk_data()

@st.cache_resource
def load_models():
    required_files = ["neural_network.pkl", "vectorizer.pkl", "label_encoder.pkl"]

    for file in required_files:
        if not os.path.exists(file):
            st.error(f"❌ Required file '{file}' not found. Please upload or place it correctly.")
            st.stop()

    model = load('neural_network.pkl')
    vectorizer = load('vectorizer.pkl')
    label_encoder = load('label_encoder.pkl')

    scaler = None
    scaling_used = False
    if os.path.exists("scaler.pkl"):
        scaler = load("scaler.pkl")
        scaling_used = True

    return model, vectorizer, label_encoder, scaler, scaling_used

model, vectorizer, label_encoder, scaler, scaling_used = load_models()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

emoji = {
    "Positive": "😃✨💖",
    "Neutral": "😐🌀🤷",
    "Negative": "😠💢👎"
}

neutral_keywords = [
    'okay', 'fine', 'average', 'meh', 'just okay', 'not that much', 'not bad',
    'mediocre', 'so-so', 'alright', 'nothing special', 'kind of', 'could be better',
    'couldn’t care less', 'indifferent', 'okay-ish', 'neither good nor bad',
    'passable', 'acceptable', 'not great', 'nothing remarkable', 'alright-ish',
    'just fine', 'could be worse', 'not bad, not good', 'somewhat okay', 'meh, could be better',
    'nothing to complain about', 'barely noticeable', 'average at best', 'mediocre at best', 'tolerable'
]

def convert_ordinals(text):
    return re.sub(r'\b(\d+)(st|nd|rd|th)\b',
                  lambda m: num2words(int(m.group(1)), to='ordinal'),
                  text)

def preprocess_review(review):
    review = str(review).lower()
    review = convert_ordinals(review)
    review = contractions.fix(review)
    review = re.sub(r"http\S+", "", review)
    review = re.sub(r'\S*\d\S*', '', review).strip()
    review = re.sub(r'[^a-zA-Z\s]', ' ', review)
    review = re.sub(r'(.)\1{2,}', r'\1\1', review)
    tokens = word_tokenize(review)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and (len(w) > 1 or w in {'no', 'ok', 'go'})]
    return ' '.join(tokens)

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

st.markdown("""
<div style='text-align: center; padding: 15px; border: 1px solid #ddd; border-radius: 10px;'>
    <h1>💬 Real-time Sentiment Classifier</h1>
    <p style='font-size:16px;'>Classify product reviews as <b style='color:green;'>Positive</b>, <b style='color:orange;'>Neutral</b>, or <b style='color:red;'>Negative</b></p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style='border: 1px solid #ddd; border-radius: 10px; padding: 15px; text-align:center;'>
    <h4>📋 Try an example</h4>
    <p style='font-size:14px;'>Click any button below to auto-feed the example in the input box.</p>
</div>
""", unsafe_allow_html=True)

col_ex1, col_ex2, col_ex3 = st.columns([2, 6, 2])
with col_ex2:
    col1, col2, col3 = st.columns(3)
    if col1.button("😃 Positive"):
        st.session_state.user_input = "Absolutely love this product! Works like a charm."
    if col2.button("😐 Neutral"):
        st.session_state.user_input = "It's okay, nothing too great or too bad."
    if col3.button("😠 Negative"):
        st.session_state.user_input = "Terrible experience. Waste of money."

st.markdown("<div style='text-align:center;'><label style='font-size:16px;font-weight:bold;'>✍️ Enter a review to classify:</label></div>", unsafe_allow_html=True)
user_input = st.text_area("", value=st.session_state.user_input, height=100, key="user_input", label_visibility="collapsed")

col_btn1, col_btn2, col_btn3 = st.columns([1.5, 2, 1.5])
with col_btn2:
    col1, col2 = st.columns(2)
    predict_clicked = col1.button("🔍 Predict", use_container_width=True)
    clear_clicked = col2.button("🧹 Reset All", use_container_width=True)

if clear_clicked:
    st.session_state.user_input = ""

if predict_clicked:
    if not user_input.strip():
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        clean_text = preprocess_review(user_input)
        tfidf_input = vectorizer.transform([clean_text])

        review_len = len(clean_text)
        word_count = len(clean_text.split())
        exclam_count = user_input.count("!")
        extra_features = [[review_len, word_count, exclam_count]]

        if scaling_used:
            extra_features = scaler.transform(extra_features)
        extra_sparse = csr_matrix(extra_features)
        final_input = hstack([tfidf_input, extra_sparse])

        probs = model.predict_proba(final_input)[0]
        prediction = model.predict(final_input)[0]

        label_classes = list(label_encoder.classes_)
        if isinstance(prediction, (int, np.integer)):
            label = label_encoder.inverse_transform([prediction])[0]
        else:
            label = prediction

        user_input_lower = user_input.lower()
        neutral_index = label_classes.index("Neutral")

        if any(keyword in user_input_lower for keyword in neutral_keywords):
            label = 'Neutral'
            confidence = 100.00
        elif probs[neutral_index] >= 0.20:
            label = 'Neutral'
            confidence = probs[neutral_index] * 100
        else:
            label_index = label_classes.index(label)
            confidence = probs[label_index] * 100

        with st.spinner("Analyzing review..."):
            time.sleep(1.5)

        st.markdown(f"""
        <div style='text-align:center; border: 1px solid #ddd; border-radius: 10px; padding: 15px;'>
            <h2 style='color:#0099ff;'>📢 Prediction Result</h2>
            <div style='font-size:22px;'>{emoji.get(label, '🔍')} Sentiment is <b>{label}</b> <span style='font-size:16px;'>(Confidence: {confidence:.2f}%)</span></div>
        </div>
        """, unsafe_allow_html=True)

        col_plot, col_meta = st.columns(2)

        with col_plot:
            try:
                fig, ax = plt.subplots(figsize=(3, 2.5))
                ax.pie(probs, labels=label_classes, autopct="%1.1f%%", colors=["#8BC34A", "#FFC107", "#FF5252"])
                ax.axis("equal")
                st.pyplot(fig)
            except:
                st.warning("⚠️ Could not render confidence pie chart.")

        with col_meta:
            st.markdown(f"""
            <div style='border: 1px solid #ddd; border-radius: 10px; padding: 10px;'>
                <h4 style='text-align:center;'>📊 Review Details</h4>
                <ul style='font-size:15px;'>
                    <li><b>Length:</b> {review_len} characters</li>
                    <li><b>Word Count:</b> {word_count}</li>
                    <li><b>Exclamation Marks:</b> {exclam_count}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        output_df = pd.DataFrame([{
            "Review": user_input,
            "Prediction": label,
            "Confidence": f"{confidence:.2f}%",
            "Length": review_len,
            "Word Count": word_count,
            "Exclamation Count": exclam_count
        }])

        col_dl1, col_dl2, col_dl3 = st.columns([2, 6, 2])
        with col_dl2:
            st.download_button("⬇️ Download Result as CSV", output_df.to_csv(index=False), file_name="review_prediction.csv", use_container_width=True)

        st.markdown("""
        <div style='text-align:center; padding-top: 10px;'>
            <span style='font-size:13px; color: gray;'>🤖 Powered by Neural Network | TF-IDF + Engineered Features</span>
        </div>
        """, unsafe_allow_html=True)

        st.balloons()

        st.markdown("""
        <style>
        canvas:has(+ div[data-testid="stNotificationContent"]) {
            display: none;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
        <style>
        div[data-testid="stDownloadButton"] > button {
            background-color: #ff4b4b;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            border: none;
            padding: 8px 16px;
            font-size: 14px;
        }
        div[data-testid="stDownloadButton"] > button:hover {
            background-color: #ff1a1a;
        }
        </style>
        """, unsafe_allow_html=True)
