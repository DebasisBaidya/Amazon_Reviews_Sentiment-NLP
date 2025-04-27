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
import emoji as em

# Set up Streamlit page configuration
st.set_page_config(page_title="Sentiment Classifier", layout="centered")

# Function to ensure necessary NLTK data is downloaded
def ensure_nltk_data():
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource)

ensure_nltk_data()

# Function to load trained models and vectorizer
@st.cache_resource
def load_models():
    required_files = ["neural_network.pkl", "vectorizer.pkl", "label_encoder.pkl"]
    for file in required_files:
        if not os.path.exists(file):
            st.error(f"❌ Required file '{file}' not found.")
            st.stop()
    # Load models and vectorizer
    model = load('neural_network.pkl')
    vectorizer = load('vectorizer.pkl')
    label_encoder = load('label_encoder.pkl')
    scaler = load("scaler.pkl") if os.path.exists("scaler.pkl") else None
    scaling_used = scaler is not None  # Check if scaling was applied during training
    return model, vectorizer, label_encoder, scaler, scaling_used

model, vectorizer, label_encoder, scaler, scaling_used = load_models()

# Define stopwords and lemmatizer for text preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Emojis for sentiment labels
emoji = {
    "Positive": "😃✨💖",
    "Neutral": "😐🌀🤷",
    "Negative": "👿💢👎"
}

# Neutral keywords for keyword-based adjustments
neutral_keywords = [
    'okay', 'fine', 'average', 'meh', 'just okay', 'not that much', 'not bad',
    'mediocre', 'so-so', 'alright', 'nothing special', 'kind of', 'could be better',
    'couldn\u2019t care less', 'indifferent', 'okay-ish', 'neither good nor bad',
    'passable', 'acceptable', 'not great', 'nothing remarkable', 'alright-ish',
    'just fine', 'could be worse', 'not bad, not good', 'somewhat okay', 'meh, could be better',
    'nothing to complain about', 'barely noticeable', 'average at best', 'mediocre at best', 'tolerable'
]

# Function to convert ordinals (1st, 2nd, 3rd) to words
def convert_ordinals(text):
    return re.sub(r'\b(\d+)(st|nd|rd|th)\b',
                  lambda m: num2words(int(m.group(1)), to='ordinal'),
                  text)

# Function to preprocess the review text
def preprocess_review(review):
    review = str(review).lower()  # Convert to lowercase
    review = convert_ordinals(review)  # Convert ordinal numbers
    review = contractions.fix(review)  # Expand contractions (like "I'm" to "I am")
    review = re.sub(r"http\S+", "", review)  # Remove URLs
    review = re.sub(r'\S*\d\S*', '', review).strip()  # Remove words with digits
    review = re.sub(r'[^a-zA-Z\s]', ' ', review)  # Remove non-alphabet characters
    review = re.sub(r'(.)\1{2,}', r'\1\1', review)  # Reduce repeated characters (like "loooove" to "love")
    tokens = word_tokenize(review)  # Tokenize the review
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and (len(w) > 1 or w in {'no', 'ok', 'go'})]  # Lemmatize and remove stopwords
    return ' '.join(tokens)  # Return processed review as a string

# Function to count emojis in the text
def count_emojis(text):
    return sum(1 for char in text if char in em.EMOJI_DATA)  # Count emoji characters

# Initialize session state for user input if not already set
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Display header
st.markdown("""  
<div style='text-align: center; padding: 15px; border: 1px solid #ddd; border-radius: 10px;'>
    <h1>💬 Real-time Sentiment Classifier</h1>
    <p style='font-size:16px;'>Classify product reviews as <b style='color:green;'>Positive</b>, <b style='color:orange;'>Neutral</b>, or <b style='color:red;'>Negative</b></p>
</div>
""", unsafe_allow_html=True)

# Add input area and buttons for prediction and reset
st.text_area("Enter a review to classify", value=st.session_state.user_input, height=100, key="user_input")
col_btn1, col_btn2 = st.columns([2, 1])
with col_btn2:
    predict_clicked = st.button("🔍 Predict", use_container_width=True)
    clear_clicked = st.button("🧹 Reset All", use_container_width=True)

# If Reset All is clicked, clear session state and rerun
if clear_clicked:
    st.session_state.user_input = ""  # Clears session state input
    st.experimental_rerun()

# Handle prediction when Predict button is clicked
if predict_clicked:
    if not st.session_state.user_input.strip():
        st.warning("⚠️ Please enter a review to analyze.")  # Show warning if input is empty
    else:
        user_input = st.session_state.user_input
        clean_text = preprocess_review(user_input)  # Preprocess the user input
        emoji_count_val = count_emojis(user_input)  # Count emojis in the review
        sentiment_score = TextBlob(clean_text).sentiment.polarity  # Calculate sentiment score using TextBlob

        # Perform the model prediction
        tfidf_input = vectorizer.transform([clean_text])  # Transform the review using the vectorizer
        review_len = len(clean_text)  # Length of the cleaned review
        word_count = len(clean_text.split())  # Word count of the review
        exclam_count = user_input.count("!")  # Count of exclamation marks
        extra_features = [[review_len, word_count, exclam_count]]  # Additional features for prediction

        if scaling_used:
            extra_features = scaler.transform(extra_features)  # Scale features if scaling was applied during training
        extra_sparse = csr_matrix(extra_features)  # Convert features to sparse matrix
        final_input = hstack([tfidf_input, extra_sparse])  # Combine TF-IDF and additional features

        probs = model.predict_proba(final_input)[0]  # Get probability predictions
        prediction = model.predict(final_input)[0]  # Get the final predicted class

        label_classes = list(label_encoder.classes_)  # Get label classes
        if isinstance(prediction, (int, np.integer)):
            label = label_encoder.inverse_transform([prediction])[0]  # Get label from encoder
        else:
            label = prediction

        # Check for neutral sentiment based on keywords or confidence threshold
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
            label_index = np.argmax(probs)
            label = label_classes[label_index]
            confidence = probs[label_index] * 100

        # Display prediction result
        st.markdown(f"""
        <div style='text-align:center; border: 1px solid #ddd; border-radius: 10px; padding: 15px;'>
            <h2 style='color:#0099ff;'>📢 Prediction Result</h2>
            <div style='font-size:22px;'>{emoji.get(label, '🔍')} Sentiment is <b>{label}</b> <span style='font-size:16px;'>(Confidence: {confidence:.2f}%)</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Display pie chart for sentiment probabilities
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

        # Allow user to download prediction results as CSV
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

        # Display balloons to celebrate
        st.balloons()
