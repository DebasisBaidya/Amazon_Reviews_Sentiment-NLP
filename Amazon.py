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
import emoji  # Importing emoji library to count emojis

# Set page config
st.set_page_config(page_title="Sentiment Classifier", layout="centered")

# Function to ensure required nltk data is available
def ensure_nltk_data():
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource)

# Ensure necessary resources are available
ensure_nltk_data()

# Function to load trained models and other necessary files
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

# Load models and other necessary files
model, vectorizer, label_encoder, scaler, scaling_used = load_models()

# Initialize nltk stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Emoji dictionary for display
emoji_dict = {
    "Positive": "😃✨💖",
    "Neutral": "😐🌀🤷",
    "Negative": "👿💢👎"
}

# Neutral keywords list
neutral_keywords = [
    'okay', 'fine', 'average', 'meh', 'just okay', 'not that much', 'not bad',
    'mediocre', 'so-so', 'alright', 'nothing special', 'kind of', 'could be better',
    'couldn’t care less', 'indifferent', 'okay-ish', 'neither good nor bad',
    'passable', 'acceptable', 'not great', 'nothing remarkable', 'alright-ish',
    'just fine', 'could be worse', 'not bad, not good', 'somewhat okay', 'meh, could be better',
    'nothing to complain about', 'barely noticeable', 'average at best', 'mediocre at best', 'tolerable'
]

# Function to convert ordinals like 1st, 2nd, 3rd to 1st, 2nd, 3rd words
def convert_ordinals(text):
    return re.sub(r'\b(\d+)(st|nd|rd|th)\b',
                  lambda m: num2words(int(m.group(1)), to='ordinal'),
                  text)

# Preprocessing function for cleaning review text
def preprocess_review(review):
    review = str(review).lower()
    review = convert_ordinals(review)  # Convert ordinal numbers (1st, 2nd, etc.)
    review = contractions.fix(review)  # Replace contractions (e.g., "I'm" -> "I am")
    review = re.sub(r"http\S+", "", review)  # Remove URLs
    review = re.sub(r'\S*\d\S*', '', review).strip()  # Remove words with numbers
    review = re.sub(r'[^a-zA-Z\s]', ' ', review)  # Keep only alphabetic characters
    review = re.sub(r'(.)\1{2,}', r'\1\1', review)  # Replace repeated characters (e.g., "sooo" -> "so")
    tokens = word_tokenize(review)  # Tokenize the review
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and (len(w) > 1 or w in {'no', 'ok', 'go'})]  # Lemmatize and remove stop words
    return ' '.join(tokens)

# Function to count the total number of emojis in the text
def analyze_emojis(text):
    # Count the emojis in the text using the emoji library
    return sum(1 for char in text if char in emoji.EMOJI_DATA)

# If the user has not entered a review, initialize the session state
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Page heading
st.markdown("""
<div style='text-align: center; padding: 15px; border: 1px solid #ddd; border-radius: 10px;'>
    <h1>💬 Real-time Sentiment Classifier</h1>
    <p style='font-size:16px;'>Classify product reviews as <b style='color:green;'>Positive</b>, <b style='color:orange;'>Neutral</b>, or <b style='color:red;'>Negative</b></p>
</div>
""", unsafe_allow_html=True)

# Example button for trying out reviews
with st.container():
    st.markdown("""
    <div style='border: 1px solid #ddd; border-radius: 10px; padding: 15px; text-align:center;'>
        <h4>📋 Try an example</h4>
        <p style='font-size:14px;'>Click any button below to auto-feed the example in the input box.</p>
    </div>
    """, unsafe_allow_html=True)

# Add space after text and above buttons
st.markdown("<br>", unsafe_allow_html=True)

col_ex1, col_ex2, col_ex3 = st.columns([2, 6, 2])
with col_ex2:
    col1, col2, col3 = st.columns(3)
    if col1.button("😃 Positive"):
        st.session_state.user_input = "Absolutely love this product! Works like a charm."
    if col2.button("😐 Neutral"):
        st.session_state.user_input = "It's okay, nothing too great or too bad."
    if col3.button("😠 Negative"):
        st.session_state.user_input = "Terrible experience. Waste of money."

# Add space after text and above buttons
st.markdown("<br>", unsafe_allow_html=True)

# User input text area for entering reviews
st.markdown("<div style='text-align:center;'><label style='font-size:16px;font-weight:bold;'>✍️ Enter a review to classify:</label></div>", unsafe_allow_html=True)
user_input = st.text_area("", value=st.session_state.user_input, height=100, key="user_input", label_visibility="collapsed")

# Prediction and reset buttons
col_btn1, col_btn2, col_btn3 = st.columns([1.5, 2, 1.5])
with col_btn2:
    col1, col2 = st.columns(2)
    predict_clicked = col1.button("🔍 Predict", use_container_width=True)
    clear_clicked = col2.button("🧹 Reset All", use_container_width=True)

# Clear input if reset button clicked
if clear_clicked:
    st.session_state.user_input = ""  # Clears session state input
    user_input = ""  # Clears the text area input box immediately

# Prediction logic when the predict button is clicked
if predict_clicked:
    if not user_input.strip():
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        clean_text = preprocess_review(user_input)  # Preprocess the user input text
        tfidf_input = vectorizer.transform([clean_text])  # Transform input with TF-IDF

        # Feature extraction for review length, word count, exclamations
        review_len = len(clean_text)
        word_count = len(clean_text.split())
        exclam_count = user_input.count("!")
        extra_features = [[review_len, word_count, exclam_count]]

        # Apply scaling if required
        if scaling_used:
            extra_features = scaler.transform(extra_features)
        extra_sparse = csr_matrix(extra_features)
        final_input = hstack([tfidf_input, extra_sparse])  # Combine TF-IDF features with extra features

        # Predict the sentiment probabilities and label
        probs = model.predict_proba(final_input)[0]
        prediction = model.predict(final_input)[0]

        label_classes = list(label_encoder.classes_)
        if isinstance(prediction, (int, np.integer)):
            label = label_encoder.inverse_transform([prediction])[0]
        else:
            label = prediction

        # Adjust for neutral keywords and probability thresholds
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
            confidence = probs[1] * 100  # Neutral confidence percentage
        else:
            confidence = probs[label_classes.index(label)] * 100

        sentiment_score = TextBlob(clean_text).sentiment.polarity  # Sentiment score with TextBlob

        # Emoji count
        emoji_count_val = analyze_emojis(user_input)  # Get emoji count in the review
        
# Side-by-side output boxes with equal width columns
col1, col2 = st.columns([1, 1])  # Ensuring both columns have equal width

# Display review analysis on the left
with col1:
    st.markdown("""
        <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px; width: 100%;'>
            <h4 style='text-align:center;'>📊 Review Analysis</h4>
    </div>
    """, unsafe_allow_html=True)
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

# Confidence breakdown with pie chart on the right
with col2:
    st.markdown("""
    <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px; width: 100%;'>
        <h4 style='text-align:center;'>📈 Confidence Breakdown</h4>
    </div>
    """, unsafe_allow_html=True)

    # Create pie chart for confidence breakdown
    fig, ax = plt.subplots()
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [probs[0], probs[1], probs[2]]
    colors = ['#28a745', '#ffc107', '#dc3545']  # Green for Positive, Yellow for Neutral, Red for Negative
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

# Download button below both columns
with st.container():
    col_dl1, col_dl2, col_dl3 = st.columns([2, 6, 2])
    with col_dl2:
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

        st.download_button("⬇️ Download Result as CSV", output_df.to_csv(index=False), file_name="review_prediction.csv", use_container_width=True)
