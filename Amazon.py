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
import matplotlib.pyplot as plt
import os
from textblob import TextBlob
import emoji

# Set page config
st.set_page_config(page_title="Sentiment Classifier", layout="centered")

# Ensure necessary nltk data
def ensure_nltk_data():
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource)

ensure_nltk_data()

# Load models
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
    scaling_used = scaler is not None
    return model, vectorizer, label_encoder, scaler, scaling_used

model, vectorizer, label_encoder, scaler, scaling_used = load_models()

# NLP utilities
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Emojis
emoji_dict = {
    "Positive": "😃✨💖",
    "Neutral": "😐🌀🤷",
    "Negative": "👿💢👎"
}

neutral_keywords = [
    'okay', 'fine', 'average', 'meh', 'just okay', 'not that much', 'not bad',
    'mediocre', 'so-so', 'alright', 'nothing special', 'kind of', 'could be better',
    'couldn’t care less', 'indifferent', 'okay-ish', 'neither good nor bad',
    'passable', 'acceptable', 'not great', 'nothing remarkable', 'alright-ish',
    'just fine', 'could be worse', 'not bad, not good', 'somewhat okay', 'meh, could be better',
    'nothing to complain about', 'barely noticeable', 'average at best', 'mediocre at best', 'tolerable'
]

# Preprocessing
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

# UI: Page Heading
st.markdown("""
<div style='text-align: center; padding: 15px; border: 1px solid #ddd; border-radius: 10px;'>
    <h1>💬 Real-time Sentiment Classifier</h1>
    <p style='font-size:16px;'>Classify product reviews as <b style='color:green;'>Positive</b>, <b style='color:orange;'>Neutral</b>, or <b style='color:red;'>Negative</b></p>
</div>
""", unsafe_allow_html=True)

# Example Buttons
with st.container():
    st.markdown("""
    <div style='border: 1px solid #ddd; border-radius: 10px; padding: 15px; text-align:center;'>
        <h4>📋 Try an example</h4>
        <p style='font-size:14px;'>Click a button to auto-fill a sample review.</p>
    </div>
    """, unsafe_allow_html=True)

col_ex1, col_ex2, col_ex3 = st.columns([2, 6, 2])
with col_ex2:
    col1, col2, col3 = st.columns(3)
    if col1.button("😃 Positive"):
        st.session_state.user_input = "Absolutely love this product! Works like a charm."
    if col2.button("😐 Neutral"):
        st.session_state.user_input = "It's okay, nothing too great or too bad."
    if col3.button("👿 Negative"):
        st.session_state.user_input = "Terrible experience. Waste of money."

# Text Input
st.markdown("<div style='text-align:center;'><label style='font-size:16px;font-weight:bold;'>✍️ Enter a review to classify:</label></div>", unsafe_allow_html=True)
user_input = st.text_area("", value=st.session_state.get("user_input", ""), height=100, key="user_input", label_visibility="collapsed")

# Buttons
col_btn1, col_btn2, col_btn3 = st.columns([1.5, 2, 1.5])
with col_btn2:
    col1, col2 = st.columns(2)
    predict_clicked = col1.button("🔍 Predict", use_container_width=True)
    clear_clicked = col2.button("🧹 Reset All", use_container_width=True)

if clear_clicked:
    st.session_state.user_input = ""
    user_input = ""

# Prediction
if predict_clicked:
    if not user_input.strip():
        st.warning("⚠️ Please enter a review.")
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

        def contains_keyword(text, keywords):
            for kw in keywords:
                if re.search(rf'\b{re.escape(kw)}\b', text):
                    return True
            return False
        
        neutral_threshold = 0.30
        user_input_lower = user_input.lower()
        neutral_keyword_present = contains_keyword(user_input_lower, neutral_keywords)

        if probs[1] >= neutral_threshold or neutral_keyword_present:
            label = 'Neutral'
            confidence = probs[1] * 100
        else:
            confidence = probs[label_classes.index(label)] * 100

        sentiment_score = TextBlob(clean_text).sentiment.polarity
        emoji_count_val = analyze_emojis(user_input)

        ### Final Result Box
        st.markdown(f"""
        <div style="padding:20px; border-radius:10px; background-color:#f0f2f6; text-align:center;">
            <h2 style="font-size:30px;">{emoji_dict[label]}<br><span style="color: {'green' if label=='Positive' else 'orange' if label=='Neutral' else 'red'};">{label}</span></h2>
            <p style="font-size:18px;">Confidence: <b>{confidence:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        # Confidence Breakdown Pie Chart
        st.markdown("<h4 style='text-align:center;'>📈 Confidence Breakdown</h4>", unsafe_allow_html=True)
        sentiments = ["Positive", "Neutral", "Negative"]
        sentiment_probs = [probs[label_classes.index('Positive')], probs[label_classes.index('Neutral')], probs[label_classes.index('Negative')]]
        colors = ['#28a745', '#ffc107', '#dc3545']

        fig, ax = plt.subplots()
        ax.pie(sentiment_probs, labels=sentiments, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')
        st.pyplot(fig)

        # Review Analysis
        st.markdown("<h4 style='text-align:center;'>📊 Review Analysis</h4>", unsafe_allow_html=True)
        st.markdown(f"""
        <ul style='font-size:16px;'>
            <li><b>📝 Length:</b> {review_len} characters</li>
            <li><b>📚 Words:</b> {word_count}</li>
            <li><b>❗❗ Exclamations:</b> {exclam_count}</li>
            <li><b>😃 Emojis:</b> {emoji_count_val}</li>
            <li><b>❤️ Sentiment Score:</b> {sentiment_score:.3f}</li>
        </ul>
        """, unsafe_allow_html=True)

        # Download result
        output_df = pd.DataFrame([{
            "Review": user_input,
            "Prediction": label,
            "Confidence": f"{confidence:.2f}%",
            "Length": review_len,
            "Word Count": word_count,
            "Exclamation Count": exclam_count,
            "Emoji Count": emoji_count_val,
            "Sentiment Score": sentiment_score
        }])

        st.download_button("⬇️ Download as CSV", output_df.to_csv(index=False), file_name="review_prediction.csv", use_container_width=True)

        st.markdown("""
        <div style='text-align:center; padding-top:10px;'>
            <span style='font-size:13px; color:gray;'>🤖 Powered by Neural Network (MLP) | TF-IDF + Extra Features</span>
        </div>
        """, unsafe_allow_html=True)
