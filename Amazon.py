import streamlit as st
from joblib import load
import re
import emoji
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack, csr_matrix
import numpy as np
import pandas as pd

# --- Basic Setup ---
st.set_page_config(page_title="Sentiment Classifier", layout="wide")

# --- Ensure nltk resources ---
def ensure_nltk_data():
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4"]
    for res in resources:
        try:
            nltk.data.find(f"tokenizers/{res}" if res == "punkt" else f"corpora/{res}")
        except LookupError:
            nltk.download(res)

ensure_nltk_data()

# --- Load models ---
@st.cache_resource
def load_models():
    model = load('neural_network.pkl')
    vectorizer = load('vectorizer.pkl')
    label_encoder = load('label_encoder.pkl')
    scaler = load("scaler.pkl") if os.path.exists("scaler.pkl") else None
    return model, vectorizer, label_encoder, scaler

model, vectorizer, label_encoder, scaler = load_models()

# --- Preprocessing ---
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r'\S*\d\S*', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def analyze_emojis(text):
    return sum(1 for char in text if char in emoji.EMOJI_DATA)

# --- Main UI ---
st.title("💬 Real-time Sentiment Classifier")

example_col = st.columns(3)
if example_col[0].button("😊 Positive"):
    st.session_state.user_input = "I love this product, it’s amazing!"
if example_col[1].button("😐 Neutral"):
    st.session_state.user_input = "It’s okay, nothing special."
if example_col[2].button("😡 Negative"):
    st.session_state.user_input = "Terrible experience, would not recommend."

st.write("---")

user_input = st.text_area("✍️ Enter your review here:", value=st.session_state.get("user_input", ""))

predict_col, reset_col = st.columns(2)
predict = predict_col.button("🔍 Predict")
reset = reset_col.button("🧹 Reset")

if reset:
    st.session_state.user_input = ""
    user_input = ""

# --- Prediction Section ---
if predict and user_input.strip():
    clean_text = preprocess(user_input)
    tfidf = vectorizer.transform([clean_text])

    # Extra features
    review_len = len(clean_text)
    word_count = len(clean_text.split())
    exclam_count = user_input.count('!')
    emoji_count = analyze_emojis(user_input)

    extra_features = [[review_len, word_count, exclam_count]]
    if scaler:
        extra_features = scaler.transform(extra_features)
    extra_sparse = csr_matrix(extra_features)

    final_input = hstack([tfidf, extra_sparse])

    probs = model.predict_proba(final_input)[0]
    pred_class = model.predict(final_input)[0]
    label = label_encoder.inverse_transform([pred_class])[0]

    confidence = probs[label_encoder.classes_.tolist().index(label)] * 100
    sentiment_score = TextBlob(clean_text).sentiment.polarity
    tone = "Positive" if sentiment_score > 0.2 else "Negative" if sentiment_score < -0.2 else "Neutral"
    likely_about = ", ".join(TextBlob(user_input).noun_phrases) or "General experience"

    # Layout for results
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("📈 Confidence Breakdown")
        fig, ax = plt.subplots()
        labels = ['Positive', 'Neutral', 'Negative']
        colors = ['green', 'yellow', 'red']
        sizes = [probs[0], probs[1], probs[2]]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=[f"{l} {s*100:.1f}%" for l, s in zip(labels, sizes)],
            colors=colors, startangle=90, autopct=None, textprops={'fontsize': 10, 'weight': 'bold'}
        )
        ax.axis('equal')
        st.pyplot(fig)

    with right_col:
        st.subheader("📊 Review Analysis")
        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence:.2f}%")
        st.write(f"**Tone:** {tone}")
        st.write(f"**Likely About:** {likely_about}")
        st.write("---")
        st.write(f"**📝 Review Length:** {review_len} characters")
        st.write(f"**📚 Word Count:** {word_count}")
        st.write(f"**❗ Exclamation Marks:** {exclam_count}")
        st.write(f"**😃 Emoji Count:** {emoji_count}")
        st.write(f"**❤️ Sentiment Score:** {sentiment_score:.3f}")

    # Download CSV
    output = pd.DataFrame([{
        "Review": user_input,
        "Prediction": label,
        "Confidence": f"{confidence:.2f}%",
        "Tone": tone,
        "Likely About": likely_about,
        "Review Length": review_len,
        "Word Count": word_count,
        "Exclamation Count": exclam_count,
        "Emoji Count": emoji_count,
        "Sentiment Score": sentiment_score
    }])

    st.download_button("⬇️ Download Result", output.to_csv(index=False), file_name="sentiment_result.csv")

# Small CSS tweaks
st.markdown("""
<style>
button {
    height: 3em;
    font-size: 16px;
}
textarea {
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)
