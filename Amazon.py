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
import emoji

# Set page config
st.set_page_config(page_title="Sentiment Classifier", layout="centered")

# Ensure NLTK resources are downloaded
def ensure_nltk_data():
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource)

ensure_nltk_data()

# Load models and vectorizer
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

# Setup NLTK processing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

emoji_map = {
    "Positive": "😃✨💖",
    "Neutral": "😐🌀🤷",
    "Negative": "👿💢👎"
}

neutral_keywords = [
    'okay', 'fine', 'average', 'meh', 'just okay', 'not that much', 'not bad',
    'mediocre', 'so-so', 'alright', 'nothing special', 'kind of', 'could be better',
    'couldn\u2019t care less', 'indifferent', 'okay-ish', 'neither good nor bad',
    'passable', 'acceptable', 'not great', 'nothing remarkable', 'alright-ish',
    'just fine', 'could be worse', 'not bad, not good', 'somewhat okay', 'meh, could be better',
    'nothing to complain about', 'barely noticeable', 'average at best', 'mediocre at best', 'tolerable'
]

# Convert ordinal numbers to words (e.g. "1st" -> "first")
def convert_ordinals(text):
    return re.sub(r'\b(\d+)(st|nd|rd|th)\b',
                  lambda m: num2words(int(m.group(1)), to='ordinal'),
                  text)

# Preprocess the review text
def preprocess_review(review):
    review = str(review).lower()  # Convert to lowercase
    review = convert_ordinals(review)  # Convert ordinal numbers to words
    review = contractions.fix(review)  # Expand contractions (e.g., "I'm" to "I am")
    review = re.sub(r"http\S+", "", review)  # Remove URLs
    review = re.sub(r'\S*\d\S*', '', review).strip()  # Remove words with digits
    review = re.sub(r'[^a-zA-Z\s]', ' ', review)  # Remove non-alphabetic characters
    review = re.sub(r'(.)\1{2,}', r'\1\1', review)  # Reduce character repetition (e.g., "loooove" -> "love")
    tokens = word_tokenize(review)  # Tokenize the review
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and (len(w) > 1 or w in {'no', 'ok', 'go'})]  # Lemmatize and remove stopwords
    return ' '.join(tokens)  # Return the processed review as a string

# Count the number of emojis in the text
def count_emojis(text):
    return sum(1 for char in text if char in emoji.EMOJI_DATA)

# Initialize session state for user input if not already set
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Display the header
st.markdown("""
<div style='text-align: center; padding: 15px; border: 1px solid #ddd; border-radius: 10px;'>
    <h1>💬 Real-time Sentiment Classifier</h1>
    <p style='font-size:16px;'>Classify product reviews as <b style='color:green;'>Positive</b>, <b style='color:orange;'>Neutral</b>, or <b style='color:red;'>Negative</b></p>
</div>
""", unsafe_allow_html=True)

# Input area and buttons for prediction
st.text_area("Enter a review to classify", value=st.session_state.user_input, height=100, key="user_input")

col_btn1, col_btn2 = st.columns([2, 1])
with col_btn2:
    predict_clicked = st.button("🔍 Predict", use_container_width=True)
    clear_clicked = st.button("🧹 Reset All", use_container_width=True)

# Clear user input if reset button is clicked
if clear_clicked:
    st.session_state.user_input = ""  # Clears session state input
    st.experimental_rerun()

# Handle prediction when Predict button is clicked
if predict_clicked:
    if not st.session_state.user_input.strip():
        st.warning("⚠️ Please enter a review to analyze.")  # Show warning if input is empty
    else:
        user_input = st.session_state.user_input
        clean_text = preprocess_review(user_input)  # Preprocess the review text
        emoji_count_val = count_emojis(user_input)  # Count emojis in the review
        sentiment_score = TextBlob(clean_text).sentiment.polarity  # Calculate sentiment score using TextBlob

        # Prepare the review for model prediction
        tfidf_input = vectorizer.transform([clean_text])  # Transform the review using the vectorizer
        review_len = len(clean_text)  # Get the length of the cleaned review
        word_count = len(clean_text.split())  # Get the word count
        exclam_count = user_input.count("!")  # Count exclamation marks
        extra_features = [[review_len, word_count, exclam_count]]  # Collect additional features

        # Apply scaling if necessary
        if scaling_used:
            extra_features = scaler.transform(extra_features)
        extra_sparse = csr_matrix(extra_features)  # Convert to sparse matrix
        final_input = hstack([tfidf_input, extra_sparse])  # Combine TF-IDF and extra features

        # Predict sentiment probabilities and final label
        probs = model.predict_proba(final_input)[0]
        prediction = model.predict(final_input)[0]
        label_classes = list(label_encoder.classes_)
        
        # Get the predicted label
        if isinstance(prediction, (int, np.integer)):
            label = label_encoder.inverse_transform([prediction])[0]
        else:
            label = prediction

        # Check for neutral sentiment based on keywords or threshold
        neutral_threshold = 0.30
        user_input_lower = user_input.lower()
        neutral_keyword_present = any(kw in user_input_lower for kw in neutral_keywords)

        if probs[1] >= neutral_threshold or neutral_keyword_present:
            label = 'Neutral'
            confidence = probs[1] * 100
        else:
            label_index = np.argmax(probs)
            label = label_classes[label_index]
            confidence = probs[label_index] * 100

        # Display prediction result with emoji and confidence
        st.markdown(f"""
        <div style='text-align:center; border: 1px solid #ddd; border-radius: 10px; padding: 15px;'>
            <h2 style='color:#0099ff;'>📢 Prediction Result</h2>
            <div style='font-size:22px;'>{emoji_map.get(label, '🔍')} Sentiment is <b>{label}</b> <span style='font-size:16px;'>(Confidence: {confidence:.2f}%)</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Display review analysis (length, word count, etc.)
        col2 = st.columns([2])[0]  # Define column for review analysis display
        with col2:
            st.markdown("""
            <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px; width: 100%;'>
                <h4 style='text-align:center;'>📊 Review Analysis</h4>
                <ul style='font-size:16px; line-height:1.8;'>
                    <li><b>📝 Review Length:</b> {review_len} characters</li>
                    <li><b>📚 Word Count:</b> {word_count}</li>
                    <li><b>❗❗ Exclamation Marks:</b> {exclam_count}</li>
                    <li><b>😃 Emoji Count:</b> {emoji_count_val}</li>
                    <li><b>❤️ Sentiment Score:</b> {sentiment_score:.3f}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Allow user to download prediction result as CSV
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

        # Display balloons to celebrate the prediction
        st.balloons()
