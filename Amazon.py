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
from textblob import TextBlob
import os

# Set page config
st.set_page_config(page_title="Sentiment Classifier", layout="centered")

# Ensure NLTK data
nltk_resources = ["punkt", "stopwords", "wordnet", "omw-1.4"]
for resource in nltk_resources:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

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

# Preprocessing tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Emojis for sentiments
emoji = {
    "Positive": "😃✨💖",
    "Neutral": "😐🌀🤷",
    "Negative": "👿💢👎"
}

# Functions for preprocessing

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

def count_emojis(text):
    import emoji
    return sum(1 for char in text if char in emoji.EMOJI_DATA)

# Session state initialization
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Page header
st.markdown("""
<div style='text-align: center; padding: 15px; border: 1px solid #ddd; border-radius: 10px;'>
    <h1>💬 Real-time Sentiment Classifier</h1>
    <p style='font-size:16px;'>Classify product reviews as <b style='color:green;'>Positive</b>, <b style='color:orange;'>Neutral</b>, or <b style='color:red;'>Negative</b></p>
</div>
""", unsafe_allow_html=True)

# Layout with two columns
col1, col2 = st.columns(2)

with col1:
    user_input = st.text_area("Enter your review:", value=st.session_state.user_input, height=200)

if user_input:
    clean_text = preprocess_review(user_input)

    review_len = len(user_input)
    word_count = len(user_input.split())
    exclam_count = user_input.count('!')
    emoji_count_val = count_emojis(user_input)

    # Prepare data for prediction
    vectorized_text = vectorizer.transform([clean_text])
    if scaling_used:
        extra_features = np.array([[review_len, word_count, exclam_count, emoji_count_val]])
        extra_features = scaler.transform(extra_features)
        input_data = hstack([vectorized_text, csr_matrix(extra_features)])
    else:
        input_data = vectorized_text

    prediction = model.predict(input_data)
    confidence = np.max(model.predict_proba(input_data)) * 100
    label = label_encoder.inverse_transform(prediction)[0]

    with col2:
        st.markdown("""
        <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px; width: 100%;'>
            <h4 style='text-align:center;'>📊 Review Analysis</h4>
        </div>
        """, unsafe_allow_html=True)

        sentiment_score = TextBlob(clean_text).sentiment.polarity

        st.markdown(f"""
        <div style='padding: 12px;'>
            <ul style='font-size:16px; line-height:1.8;'>
                <li><b>📝 Review Length:</b> {review_len} characters</li>
                <li><b>📚 Word Count:</b> {word_count}</li>
                <li><b>❗❗ Exclamation Marks:</b> {exclam_count}</li>
                <li><b>😃 Emoji Count:</b> {emoji_count_val}</li>
                <li><b>❤️ Sentiment Score:</b> {sentiment_score:.3f}</li>
                <li><b>🔮 Predicted Sentiment:</b> {label} {emoji[label]}</li>
                <li><b>✅ Confidence:</b> {confidence:.2f}%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Create output DataFrame
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

    # Download button
    col_dl1, col_dl2, col_dl3 = st.columns([2, 6, 2])
    with col_dl2:
        st.download_button("⬇️ Download Result as CSV", output_df.to_csv(index=False), file_name="review_prediction.csv", use_container_width=True)

    st.balloons()

# Footer
st.markdown("""
<div style='text-align:center; padding-top: 10px;'>
    <span style='font-size:13px; color: gray;'>🤖 Powered by Neural Network | TF-IDF + Engineered Features</span>
</div>
""", unsafe_allow_html=True)

# Hide Streamlit notifications
st.markdown("""
<style>
canvas:has(+ div[data-testid="stNotificationContent"]) {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# Styling for download button
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
