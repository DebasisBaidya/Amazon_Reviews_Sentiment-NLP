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
from textblob import TextBlob
import emoji
import os
import io

# Set Streamlit page config
st.set_page_config(page_title="Sentiment Classifier", layout="centered")

# Ensure NLTK resources
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

# NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

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
    return re.sub(r'\b(\d+)(st|nd|rd|th)\b', lambda m: num2words(int(m.group(1)), to='ordinal'), text)

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

# Handle Neutral Keywords and Confidence Calculation
def handle_neutral_keywords(text, probs, neutral_keywords, confidence_threshold=0.30):
    """
    Modify neutral sentiment detection by lowering the threshold and improving keyword matching.
    """
    # Check for any neutral keyword in the review
    neutral_found = any(re.search(rf'\b{re.escape(kw)}\b', text.lower()) for kw in neutral_keywords)
    
    # If the neutral keywords are found, predict Neutral with 100% confidence
    if neutral_found:
        return 'Neutral', 100.0  # Return neutral with 100% confidence if the keyword matches
    elif probs[1] >= confidence_threshold:
        return 'Neutral', probs[1] * 100  # Otherwise, fall back on model confidence
    else:
        return None, None

# Header
st.markdown(""" 
<div style='text-align: center; padding: 15px; border: 1px solid #ddd; border-radius: 10px;'> 
    <h1>💬 Real-time Sentiment Classifier</h1> 
    <p style='font-size:16px;'>Classify product reviews as <b style='color:green;'>Positive</b>, <b style='color:orange;'>Neutral</b>, or <b style='color:red;'>Negative</b></p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Try Example Button Section (Retaining original styling)
st.markdown("""
<hr>
<div style="text-align: center;">
    <h3>🎯 Try an Example Review</h3>
</div>
""", unsafe_allow_html=True)

ol_ex1, col_ex2, col_ex3 = st.columns([2, 6, 2])
with col_ex2:
    col1, col2, col3 = st.columns(3)
    if col1.button("😃 Positive"):
        st.session_state.user_input = "Absolutely love this product! Works like a charm."
    if col2.button("😐 Neutral"):
        st.session_state.user_input = "It's okay, nothing too great or too bad."
    if col3.button("😈 Negative"):
        st.session_state.user_input = "Terrible experience. Waste of money."

# Text input for user to enter review
st.markdown("<div style='text-align:center;'><label style='font-size:16px;font-weight:bold;'>✍️ Enter a review to classify:</label></div>", unsafe_allow_html=True)
user_input = st.text_area("", value=st.session_state.user_input, height=100, key="user_input", label_visibility="collapsed")

# Buttons for prediction and reset
col_left, col_center, col_right = st.columns([1.5, 2, 1.5])
with col_center:
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

        # Get probabilities and prediction
        probs = model.predict_proba(final_input)[0]
        prediction = model.predict(final_input)[0]

        label_classes = list(label_encoder.classes_)
        label = label_encoder.inverse_transform([prediction])[0] if isinstance(prediction, (int, np.integer)) else prediction

        # Handle Neutral Keyword and Confidence
        label, confidence = handle_neutral_keywords(user_input, probs, neutral_keywords)

        if label is None:  # If neutral wasn't selected by keywords, use the class with highest probability
            confidence = probs[label_classes.index(label)] * 100

        sentiment_score = TextBlob(clean_text).sentiment.polarity
        emoji_count_val = analyze_emojis(user_input)

        if label == "Positive":
            st.balloons()

        # -- Prediction result
        st.markdown(f"""
        <div style='text-align:center; border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px auto; max-width: 600px;'>
            <h2 style='color:#0099ff;'>📢 Prediction Result</h2>
            <div style='font-size:20px; color:{"green" if label == "Positive" else "orange" if label == "Neutral" else "red"};'>
                {"😃 <b>Positive</b>" if label == "Positive" else "😐 <b>Neutral</b>" if label == "Neutral" else "👿 <b>Negative</b>"} <span style='font-size:16px;'>(Confidence: {confidence:.2f}%)</span>
            </div>
            <div style='margin-top: 5px;'>{'✅ Positive review' if label == "Positive" else '🌀 Neutral review' if label == "Neutral" else '⚠️ Negative review'}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence Breakdown Section
        st.markdown("<br><div style='text-align:center;'><h3>📊 Confidence Breakdown</h3></div>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6, 4))  # Adjusted size for better visibility

        # Sentiments and their corresponding probabilities
        sentiments = ["Positive", "Neutral", "Negative"]
        sentiment_probs = [probs[0], probs[1], probs[2]]

        # Plotting the pie chart
        ax.pie(sentiment_probs, labels=sentiments, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ffcc66', '#ff6666'])
        ax.axis('equal')  # Equal aspect ratio ensures pie chart is circular.

        # Show the plot in Streamlit
        st.pyplot(fig)

        # Review Analysis Section
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
