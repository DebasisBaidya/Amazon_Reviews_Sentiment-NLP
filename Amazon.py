import streamlit as st
from joblib import load
import re
import contractions
from num2words import num2words
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os
from textblob import TextBlob

# Set page config to customize the app's title and layout
st.set_page_config(page_title="Sentiment Classifier", layout="centered")

# Function to ensure necessary NLTK data is downloaded
def ensure_nltk_data():
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]  # Define the required NLTK resources
    for resource in resources:
        try:
            # Try to load each resource
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            # If not found, download the required resource
            nltk.download(resource)

ensure_nltk_data()  # Ensure all necessary NLTK resources are available

# Load models and vectorizer (cached for performance)
@st.cache_resource
def load_models():
    # List of necessary files to load for model, vectorizer, and label encoder
    required_files = ["neural_network.pkl", "vectorizer.pkl", "label_encoder.pkl"]
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

model, vectorizer, label_encoder, scaler, scaling_used = load_models()  # Load models into memory

# Set stop words and lemmatizer for preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Define emojis for sentiment visualization
emoji = {
    "Positive": "😃✨💖",
    "Neutral": "😐🌀🤷",
    "Negative": "👿💢👎"
}

# Keywords that will be considered neutral
neutral_keywords = [
    'okay', 'fine', 'average', 'meh', 'just okay', 'not that much', 'not bad',
    'mediocre', 'so-so', 'alright', 'nothing special', 'kind of', 'could be better',
    'couldn\u2019t care less', 'indifferent', 'okay-ish', 'neither good nor bad',
    'passable', 'acceptable', 'not great', 'nothing remarkable', 'alright-ish',
    'just fine', 'could be worse', 'not bad, not good', 'somewhat okay', 'meh, could be better',
    'nothing to complain about', 'barely noticeable', 'average at best', 'mediocre at best', 'tolerable'
]

# Convert ordinal numbers to words (1st -> first)
def convert_ordinals(text):
    return re.sub(r'\b(\d+)(st|nd|rd|th)\b',
                  lambda m: num2words(int(m.group(1)), to='ordinal'),
                  text)

# Function to preprocess the review: lowercase, remove unwanted characters, lemmatize
def preprocess_review(review):
    review = str(review).lower()  # Convert review to lowercase
    review = convert_ordinals(review)  # Convert ordinals like 1st, 2nd, etc., to words
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
    import emoji
    return sum(1 for char in text if char in emoji.EMOJI_DATA)

# Initialize the session state for user input if not already present
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Custom header for the app
st.markdown("""
<div style='text-align: center; padding: 15px; border: 1px solid #ddd; border-radius: 10px;'>
    <h1>💬 Real-time Sentiment Classifier</h1>
    <p style='font-size:16px;'>Classify product reviews as <b style='color:green;'>Positive</b>, <b style='color:orange;'>Neutral</b>, or <b style='color:red;'>Negative</b></p>
</div>
""", unsafe_allow_html=True)

# Create two columns for layout: one for the sentiment prediction and the other for the analysis
col1, col2 = st.columns([3, 2])

# User input for review classification
user_input = st.text_area("Enter your review:", "", max_chars=500)

# Function to analyze review: clean, tokenize, and extract metrics
def analyze_review(review):
    cleaned_review = preprocess_review(review)  # Clean and preprocess the review
    sentiment_score = TextBlob(cleaned_review).sentiment.polarity  # Get sentiment score using TextBlob
    review_len = len(review)  # Get the length of the review
    word_count = len(review.split())  # Get the word count of the review
    exclam_count = review.count('!')  # Count exclamation marks
    emoji_count_val = count_emojis(review)  # Count emojis in the review
    return cleaned_review, sentiment_score, review_len, word_count, exclam_count, emoji_count_val

# Display the analysis when the user submits a review
if user_input:
    clean_text, sentiment_score, review_len, word_count, exclam_count, emoji_count_val = analyze_review(user_input)

    # Left column - Sentiment Classification (Main content)
    with col1:
        st.markdown("""
        <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px; width: 100%;'>
            <h4 style='text-align:center;'>📊 Review Sentiment</h4>
        </div>
        """, unsafe_allow_html=True)

        # Predict sentiment using the loaded model
        transformed_input = vectorizer.transform([clean_text])  # Transform input using the TF-IDF vectorizer
        if scaling_used:
            transformed_input = scaler.transform(transformed_input)  # Apply scaling if necessary
        prediction = model.predict(transformed_input)  # Predict sentiment
        predicted_label = label_encoder.inverse_transform(prediction)[0]  # Decode prediction

        # Display the predicted sentiment with an emoji
        st.markdown(f"**Prediction:** {predicted_label} {emoji.get(predicted_label, '')}")

    # Right column - Review Analysis (Metrics)
    with col2:
        st.markdown("""
        <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px; width: 100%;'>
            <h4 style='text-align:center;'>📊 Review Analysis</h4>
        </div>
        """, unsafe_allow_html=True)

        # Display review metrics
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
