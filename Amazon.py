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
from textblob import TextBlob

# Set page config
st.set_page_config(page_title="Sentiment Classifier", layout="centered")

# Function to ensure necessary NLTK data is available
def ensure_nltk_data():
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]  # List of required NLTK resources
    for resource in resources:
        try:
            # Try to load each resource
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            # If not found, download the required resource
            nltk.download(resource)

ensure_nltk_data()  # Ensure all necessary NLTK resources are available

# Function to load model, vectorizer, label encoder, and scaler
@st.cache_resource
def load_models():
    required_files = ["neural_network.pkl", "vectorizer.pkl", "label_encoder.pkl"]  # List of required files
    for file in required_files:
        if not os.path.exists(file):
            st.error(f"❌ Required file '{file}' not found.")  # Show error if any file is missing
            st.stop()  # Stop execution if any file is missing
    # Load the model, vectorizer, label encoder, and scaler
    model = load('neural_network.pkl')
    vectorizer = load('vectorizer.pkl')
    label_encoder = load('label_encoder.pkl')
    scaler = load("scaler.pkl") if os.path.exists("scaler.pkl") else None
    scaling_used = scaler is not None  # Check if scaling was used
    return model, vectorizer, label_encoder, scaler, scaling_used

# Load models into memory
model, vectorizer, label_encoder, scaler, scaling_used = load_models()

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Emoji mapping for sentiment
emoji = {
    "Positive": "😃✨💖",
    "Neutral": "😐🌀🤷",
    "Negative": "👿💢👎"
}

# Keywords considered neutral for prediction adjustment
neutral_keywords = [
    'okay', 'fine', 'average', 'meh', 'just okay', 'not that much', 'not bad',
    'mediocre', 'so-so', 'alright', 'nothing special', 'kind of', 'could be better',
    'couldn\u2019t care less', 'indifferent', 'okay-ish', 'neither good nor bad',
    'passable', 'acceptable', 'not great', 'nothing remarkable', 'alright-ish',
    'just fine', 'could be worse', 'not bad, not good', 'somewhat okay', 'meh, could be better',
    'nothing to complain about', 'barely noticeable', 'average at best', 'mediocre at best', 'tolerable'
]

# Function to convert ordinal numbers (1st, 2nd, etc.) to words
def convert_ordinals(text):
    return re.sub(r'\b(\d+)(st|nd|rd|th)\b',
                  lambda m: num2words(int(m.group(1)), to='ordinal'),
                  text)

# Preprocessing function for reviews
def preprocess_review(review):
    review = str(review).lower()  # Convert review to lowercase
    review = convert_ordinals(review)  # Convert ordinals like 1st, 2nd, etc. to words
    review = contractions.fix(review)  # Expand contractions (e.g., "don't" -> "do not")
    review = re.sub(r"http\S+", "", review)  # Remove any URLs
    review = re.sub(r'\S*\d\S*', '', review).strip()  # Remove any numbers
    review = re.sub(r'[^a-zA-Z\s]', ' ', review)  # Remove non-alphabetic characters
    review = re.sub(r'(.)\1{2,}', r'\1\1', review)  # Reduce repeated characters (e.g., "sooo" -> "so")
    tokens = word_tokenize(review)  # Tokenize the review text into words
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and (len(w) > 1 or w in {'no', 'ok', 'go'})]
    return ' '.join(tokens)  # Join the tokens back into a string

# Function to count the number of emojis in the text
def count_emojis(text):
    import emoji  # Import emoji library
    return sum(1 for char in text if char in emoji.EMOJI_DATA)

# Initialize session state for user input
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Custom header for the app
st.markdown("""
<div style='text-align: center; padding: 15px; border: 1px solid #ddd; border-radius: 10px;'>
    <h1>💬 Real-time Sentiment Classifier</h1>
    <p style='font-size:16px;'>Classify product reviews as <b style='color:green;'>Positive</b>, <b style='color:orange;'>Neutral</b>, or <b style='color:red;'>Negative</b></p>
</div>
""", unsafe_allow_html=True)

# Columns for layout (Review Analysis on the right)
col1, col2 = st.columns([3, 2])

# Right column - Review Analysis
with col2:
    st.markdown("""
    <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px; width: 100%;'>
        <h4 style='text-align:center;'>📊 Review Analysis</h4>
    </div>
    """, unsafe_allow_html=True)

    sentiment_score = TextBlob(clean_text).sentiment.polarity  # Sentiment analysis using TextBlob

    # Display analysis metrics
    st.markdown(f"""
    <div style='padding: 12px;'>
        <ul style='font-size:16px; line-height:1.8;'>
            <li><b>📝 Review Length:</b> {review_len} characters</li>
            <li><b>📚 Word Count:</b> {word_count}</li>
            <li><b>❗❗ Exclamation Marks:</b> {exclam_count}</li>
            <li><b>😃 Emoji Count:</b> {emoji_count_val}</li>
            <li><b>❤️ Sentiment Score:</b> {sentiment_score:.3f}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Creating the DataFrame for the result
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

# Download button for the result CSV
col_dl1, col_dl2, col_dl3 = st.columns([2, 6, 2])
with col_dl2:
    st.download_button("⬇️ Download Result as CSV", output_df.to_csv(index=False), file_name="review_prediction.csv", use_container_width=True)

# Footer with the app's information
st.markdown("""
<div style='text-align:center; padding-top: 10px;'>
    <span style='font-size:13px; color: gray;'>🤖 Powered by Neural Network | TF-IDF + Engineered Features</span>
</div>
""", unsafe_allow_html=True)

# Balloons animation for fun
st.balloons()

# Hide notification content (styling trick)
st.markdown("""
<style>
canvas:has(+ div[data-testid="stNotificationContent"]) {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# Styling for the download button
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
