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
from textblob import TextBlob
import emoji
import os

# --- Page config ---
st.set_page_config(page_title="Sentiment Classifier", layout="centered")

# --- Ensure necessary nltk data ---
def ensure_nltk_data():
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource)

ensure_nltk_data()

# --- Load models ---
@st.cache_resource
def load_models():
    required_files = ["neural_network.pkl", "vectorizer.pkl", "label_encoder.pkl"]
    for file in required_files:
        if not os.path.exists(file):
            st.error(f"❌ Required file '{file}' not found.")
            st.stop()
    model = load('neural_network.pkl')
    vectorizer = load('vectorizer.pkl')
    label_encoder = load('label_encoder.pkl')
    scaler = load("scaler.pkl") if os.path.exists("scaler.pkl") else None
    return model, vectorizer, label_encoder, scaler

model, vectorizer, label_encoder, scaler = load_models()

# --- Preprocessing functions ---
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

neutral_keywords = [
    'okay', 'fine', 'average', 'meh', 'just okay', 'not that much', 'not bad',
    'mediocre', 'so-so', 'alright', 'nothing special', 'kind of', 'could be better'
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

def analyze_emojis(text):
    return sum(1 for char in text if char in emoji.EMOJI_DATA)

def contains_keyword(text, keywords):
    for kw in keywords:
        if re.search(rf'\b{re.escape(kw)}\b', text):
            return True
    return False

# --- Session State for input ---
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# --- Main Heading ---
st.markdown("""
<div style='text-align: center; padding: 15px; border: 1px solid #ddd; border-radius: 10px;'>
    <h1>💬 Real-time Sentiment Classifier</h1>
    <p style='font-size:16px;'>Classify product reviews as <b style='color:green;'>Positive</b>, 
    <b style='color:orange;'>Neutral</b>, or <b style='color:red;'>Negative</b></p>
</div>
""", unsafe_allow_html=True)

# --- Example Reviews ---
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

st.markdown("<br>", unsafe_allow_html=True)

# --- User Input ---
st.markdown("<div style='text-align:center;'><label style='font-size:16px;font-weight:bold;'>✍️ Enter a review to classify:</label></div>", unsafe_allow_html=True)
user_input = st.text_area("", value=st.session_state.user_input, height=100, key="user_input", label_visibility="collapsed")

# --- Predict and Reset Buttons ---
col_btn1, col_btn2, col_btn3 = st.columns([1.5, 2, 1.5])
with col_btn2:
    col1, col2 = st.columns(2)
    predict_clicked = col1.button("🔍 Predict", use_container_width=True)
    clear_clicked = col2.button("🧹 Reset All", use_container_width=True)

if clear_clicked:
    st.session_state.user_input = ""
    user_input = ""

# --- Prediction Logic ---
if predict_clicked:
    if not user_input.strip():
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        clean_text = preprocess_review(user_input)
        tfidf_input = vectorizer.transform([clean_text])

        review_len = len(clean_text)
        word_count = len(clean_text.split())
        exclam_count = user_input.count("!")
        emoji_count_val = analyze_emojis(user_input)

        extra_features = [[review_len, word_count, exclam_count]]
        if scaler:
            extra_features = scaler.transform(extra_features)
        extra_sparse = csr_matrix(extra_features)
        final_input = hstack([tfidf_input, extra_sparse])

        probs = model.predict_proba(final_input)[0]
        prediction = model.predict(final_input)[0]
        label_classes = list(label_encoder.classes_)
        label = label_encoder.inverse_transform([prediction])[0] if isinstance(prediction, (int, np.integer)) else prediction

        user_input_lower = user_input.lower()
        neutral_keyword_present = contains_keyword(user_input_lower, neutral_keywords)
        if probs[1] >= 0.30 or neutral_keyword_present:
            label = 'Neutral'

        confidence = probs[label_classes.index(label)] * 100
        sentiment_score = TextBlob(clean_text).sentiment.polarity

        # Define tone
        tone = "Positive" if sentiment_score > 0.2 else "Negative" if sentiment_score < -0.2 else "Neutral"

        # Guess likely about
        blob = TextBlob(user_input)
        likely_about = ", ".join(blob.noun_phrases) if blob.noun_phrases else "General Product Experience"

        # --- Results Display ---
        col_left, col_right = st.columns(2)

        # --- Confidence Breakdown (Left) ---
        with col_left:
            st.markdown("""
            <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px;'>
                <h4 style='text-align:center;'>📈 Confidence Breakdown</h4>
            </div>
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots()
            labels = ['Positive', 'Neutral', 'Negative']
            sizes = [probs[0], probs[1], probs[2]]
            colors = ['#28a745', '#ffc107', '#dc3545']
            wedges, texts, autotexts = ax.pie(
                sizes, labels=[f"{l} {s*100:.1f}%" for l, s in zip(labels, sizes)],
                colors=colors, startangle=90, textprops=dict(color="black", weight='bold'),
                autopct=None
            )
            ax.axis('equal')
            st.pyplot(fig)

        # --- Review Analysis (Right) ---
        with col_right:
            st.markdown("""
            <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px;'>
                <h4 style='text-align:center;'>📊 Review Analysis</h4>
            </div>
            """, unsafe_allow_html=True)
            st.write(f"**Prediction:** `{label}`")
            st.write(f"**Confidence:** `{confidence:.2f}%`")
            st.write(f"**Tone:** `{tone}`")
            st.write(f"**Likely About:** `{likely_about}`")
            st.write("---")
            st.write(f"**📝 Review Length:** {review_len} characters")
            st.write(f"**📚 Word Count:** {word_count}")
            st.write(f"**❗❗ Exclamation Marks:** {exclam_count}")
            st.write(f"**😃 Emoji Count:** {emoji_count_val}")
            st.write(f"**❤️ Sentiment Score:** {sentiment_score:.3f}")

        # --- Download Button ---
        output_df = pd.DataFrame([{
            "Review": user_input,
            "Prediction": label,
            "Confidence": f"{confidence:.2f}%",
            "Tone": tone,
            "Likely About": likely_about,
            "Length": review_len,
            "Word Count": word_count,
            "Exclamation Count": exclam_count,
            "Emoji Count": emoji_count_val,
            "Sentiment Score": sentiment_score
        }])
        st.markdown("<br>", unsafe_allow_html=True)
        col_dl1, col_dl2, col_dl3 = st.columns([2, 6, 2])
        with col_dl2:
            st.download_button("⬇️ Download Result as CSV", output_df.to_csv(index=False), file_name="review_prediction.csv", use_container_width=True)

        st.balloons()

# --- Custom Styling ---
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
