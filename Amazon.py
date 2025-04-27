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

# Set page config for Streamlit
st.set_page_config(page_title="Sentiment Classifier", layout="centered")

# Ensure that necessary NLTK resources are downloaded
def ensure_nltk_data():
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource)

ensure_nltk_data()

# Function to load pre-trained models and necessary files
@st.cache_resource
def load_models():
    required_files = ["neural_network.pkl", "vectorizer.pkl", "label_encoder.pkl"]
    for file in required_files:
        if not os.path.exists(file):
            st.error(f"❌ Required file '{file}' not found.")
            st.stop()  # Stop execution if any file is missing
    model = load('neural_network.pkl')  # Load the trained neural network model
    vectorizer = load('vectorizer.pkl')  # Load the TF-IDF vectorizer
    label_encoder = load('label_encoder.pkl')  # Load the label encoder
    scaler = load("scaler.pkl") if os.path.exists("scaler.pkl") else None  # Check if scaling is used
    scaling_used = scaler is not None
    return model, vectorizer, label_encoder, scaler, scaling_used

# Load the models and other files
model, vectorizer, label_encoder, scaler, scaling_used = load_models()

# Set stopwords and lemmatizer for text preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Emojis for sentiment labels
emoji = {
    "Positive": "😃✨💖",
    "Neutral": "😐🌀🤷",
    "Negative": "👿💢👎"
}

# Keywords that can indicate a neutral sentiment
neutral_keywords = [
    'okay', 'fine', 'average', 'meh', 'just okay', 'not that much', 'not bad',
    'mediocre', 'so-so', 'alright', 'nothing special', 'kind of', 'could be better',
    'couldn\u2019t care less', 'indifferent', 'okay-ish', 'neither good nor bad',
    'passable', 'acceptable', 'not great', 'nothing remarkable', 'alright-ish',
    'just fine', 'could be worse', 'not bad, not good', 'somewhat okay', 'meh, could be better',
    'nothing to complain about', 'barely noticeable', 'average at best', 'mediocre at best', 'tolerable'
]

# Function to convert ordinal numbers to words (e.g., 1st -> first)
def convert_ordinals(text):
    return re.sub(r'\b(\d+)(st|nd|rd|th)\b',
                  lambda m: num2words(int(m.group(1)), to='ordinal'),
                  text)

# Function to preprocess the review text
def preprocess_review(review):
    review = str(review).lower()  # Convert to lowercase
    review = convert_ordinals(review)  # Convert ordinals like 1st to first
    review = contractions.fix(review)  # Expand contractions (e.g., don't -> do not)
    review = re.sub(r"http\S+", "", review)  # Remove URLs
    review = re.sub(r'\S*\d\S*', '', review).strip()  # Remove words with numbers
    review = re.sub(r'[^a-zA-Z\s]', ' ', review)  # Keep only letters and spaces
    review = re.sub(r'(.)\1{2,}', r'\1\1', review)  # Fix repeated characters (e.g., "loooove" -> "love")
    tokens = word_tokenize(review)  # Tokenize the review into words
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and (len(w) > 1 or w in {'no', 'ok', 'go'})]  # Lemmatize and remove stopwords
    return ' '.join(tokens)  # Return the cleaned and tokenized text

# Function to count emojis in the text
def count_emojis(text):
    import emoji  # Import emoji library to check for emojis in text
    return sum(1 for char in text if char in emoji.EMOJI_DATA)

# Store user input in session state
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Display the page title and description
st.markdown("""
<div style='text-align: center; padding: 15px; border: 1px solid #ddd; border-radius: 10px;'>
    <h1>💬 Real-time Sentiment Classifier</h1>
    <p style='font-size:16px;'>Classify product reviews as <b style='color:green;'>Positive</b>, <b style='color:orange;'>Neutral</b>, or <b style='color:red;'>Negative</b></p>
</div>
""", unsafe_allow_html=True)

# Input section for the user to enter a review
user_input = st.text_area("Enter your review:")

if user_input:
    # Preprocess the user input
    clean_text = preprocess_review(user_input)

    # Calculate sentiment score using TextBlob
    sentiment_score = TextBlob(clean_text).sentiment.polarity

    # Calculate other review analysis metrics
    review_len = len(user_input)  # Length of the review
    word_count = len(user_input.split())  # Word count
    exclam_count = user_input.count("!")  # Count exclamation marks
    emoji_count_val = count_emojis(user_input)  # Count emojis in the review

    # Predict sentiment using the trained model
    prediction = model.predict(vectorizer.transform([clean_text]))
    label = label_encoder.inverse_transform(prediction)[0]  # Get the label for the prediction
    confidence = model.predict_proba(vectorizer.transform([clean_text]))[0].max() * 100  # Get the confidence percentage

    # Layout for displaying results
    col1, col2 = st.columns(2)

    with col2:
        st.markdown("""<div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px; width: 100%;'>
            <h4 style='text-align:center;'>📊 Review Analysis</h4></div>""", unsafe_allow_html=True)

        # Display the review analysis metrics
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

        # Prepare data for CSV download
        output_df = pd.DataFrame([{
            "Review": user_input,
            "Prediction": label,
            "Confidence": f"{confidence:.2f}%",
            "Length": review_len,
            "Word Count": word_count,
            "Exclamation Count": exclam_count,
            "Emoji Count": emoji_count_val
        }])

        # Download button for the user to download results as CSV
        col_dl1, col_dl2, col_dl3 = st.columns([2, 6, 2])
        with col_dl2:
            st.download_button("⬇️ Download Result as CSV", output_df.to_csv(index=False), file_name="review_prediction.csv", use_container_width=True)

        # Display footer
        st.markdown("""<div style='text-align:center; padding-top: 10px;'>
            <span style='font-size:13px; color: gray;'>🤖 Powered by Neural Network | TF-IDF + Engineered Features</span>
        </div>""", unsafe_allow_html=True)

        # Trigger balloons animation for fun
        st.balloons()

        # Remove default canvas notifications
        st.markdown("""
        <style>
        canvas:has(+ div[data-testid="stNotificationContent"]) {
            display: none;
        }
        </style>
        """, unsafe_allow_html=True)

        # Custom styling for download button
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
