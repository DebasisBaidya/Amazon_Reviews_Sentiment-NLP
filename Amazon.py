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
import time

# ─── Page Config & NLTK Setup ─────────────────────────────────────────────
st.set_page_config(page_title="Sentiment Classifier", layout="centered")

def ensure_nltk_data():
    for res in ["punkt","stopwords","wordnet","omw-1.4"]:
        try:
            nltk.data.find(f"tokenizers/{res}" if res=="punkt" else f"corpora/{res}")
        except LookupError:
            nltk.download(res)
ensure_nltk_data()

# ─── Load Models ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    files = ["neural_network.pkl","vectorizer.pkl","label_encoder.pkl"]
    for f in files:
        if not os.path.exists(f):
            st.error(f"❌ Required file '{f}' not found."); st.stop()
    m = load("neural_network.pkl")
    v = load("vectorizer.pkl")
    le = load("label_encoder.pkl")
    sc = load("scaler.pkl") if os.path.exists("scaler.pkl") else None
    return m, v, le, sc, (sc is not None)
model, vectorizer, label_encoder, scaler, scaling_used = load_models()

# ─── NLP Utilities ────────────────────────────────────────────────────────
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
emoji_dict = {
    "Positive":"😃✨💖","Neutral":"😐🌀🤷","Negative":"👿💢👎"
}
neutral_kw = [
    'okay','fine','average','meh','just okay','not that much','not bad',
    'mediocre','so-so','alright','nothing special','kind of','could be better',
    'couldn’t care less','indifferent','okay-ish','neither good nor bad',
    'passable','acceptable','not great','nothing remarkable','alright-ish',
    'just fine','could be worse','not bad, not good','somewhat okay','meh, could be better',
    'nothing to complain about','barely noticeable','average at best','mediocre at best','tolerable'
]

def convert_ordinals(text):
    return re.sub(r'\b(\d+)(st|nd|rd|th)\b',
                  lambda m: num2words(int(m.group(1)), to='ordinal'), text)

def preprocess_review(txt):
    t = str(txt).lower()
    t = convert_ordinals(t)
    t = contractions.fix(t)
    t = re.sub(r"http\S+","", t)
    t = re.sub(r'\S*\d\S*','', t).strip()
    t = re.sub(r'[^a-zA-Z\s]',' ', t)
    t = re.sub(r'(.)\1{2,}', r'\1\1', t)
    toks = word_tokenize(t)
    toks = [lemmatizer.lemmatize(w) for w in toks
            if w not in stop_words and (len(w)>1 or w in {'no','ok','go'})]
    return " ".join(toks)

def analyze_emojis(txt):
    return sum(1 for c in txt if c in emoji.EMOJI_DATA)

# ─── Header & Examples ────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:15px; border:1px solid #ddd; border-radius:10px;'>
  <h1>💬 Real-time Sentiment Classifier</h1>
  <p style='font-size:16px;'>Classify product reviews as 
     <b style='color:green;'>Positive</b>, 
     <b style='color:orange;'>Neutral</b>, 
     <b style='color:red;'>Negative</b>
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style='border:1px solid #ddd; border-radius:10px; padding:15px; text-align:center;'>
  <h4>📋 Try an example</h4>
  <p style='font-size:14px;'>Click a button to auto-fill an example review.</p>
</div>
""", unsafe_allow_html=True)

# — GAP after ‘Try an example’
st.markdown("<br>", unsafe_allow_html=True)

col_e1, col_e2, col_e3 = st.columns([2,6,2])
with col_e2:
    c1, c2, c3 = st.columns(3)
    if c1.button("😃 Positive"): st.session_state.user_input="Absolutely love this product! Works like a charm."
    if c2.button("😐 Neutral"):  st.session_state.user_input="It's okay, nothing too great or too bad."
    if c3.button("👿 Negative"): st.session_state.user_input="Terrible experience. Waste of money."

# ─── Input & Buttons ─────────────────────────────────────────────────────
st.markdown("<div style='text-align:center;'><label style='font-size:16px; font-weight:bold;'>✍️ Enter a review:</label></div>", unsafe_allow_html=True)
user_input = st.text_area("", value=st.session_state.get("user_input",""), height=100, key="user_input", label_visibility="collapsed")

c_left, c_mid, c_right = st.columns([1.5,2,1.5])
with c_mid:
    b1, b2 = st.columns(2)
    predict = b1.button("🔍 Predict", use_container_width=True)
    reset   = b2.button("🧹 Reset All", use_container_width=True)
if reset:
    st.session_state.user_input = ""

# ─── Prediction ──────────────────────────────────────────────────────────
if predict:
    if not user_input.strip():
        st.warning("⚠️ Please enter a review.")
    else:
        clean = preprocess_review(user_input)
        tfidf = vectorizer.transform([clean])
        L = len(clean); W = len(clean.split()); E = user_input.count("!")
        feats = [[L,W,E]]
        if scaling_used: feats = scaler.transform(feats)
        inp = hstack([tfidf, csr_matrix(feats)])
        probs = model.predict_proba(inp)[0]
        pred = model.predict(inp)[0]
        cls = list(label_encoder.classes_)
        label = label_encoder.inverse_transform([pred])[0] if isinstance(pred,(int,np.integer)) else pred

        # force neutral if keyword or threshold
        if probs[1]>=0.30 or any(re.search(rf'\b{re.escape(k)}\b',user_input.lower()) for k in neutral_kw):
            label, conf = 'Neutral', probs[1]*100
        else:
            conf = probs[cls.index(label)]*100

        score = TextBlob(clean).sentiment.polarity
        emo_ct = analyze_emojis(user_input)
        # balloons if positive
        if label=="Positive": st.balloons()

        # ─── Prediction Card
        st.markdown(f"""
        <div style='text-align:center; border:1px solid #ddd; border-radius:10px; padding:15px; margin:10px auto; max-width:600px;'>
          <h2 style='color:#0099ff;'>🔮 Prediction Result</h2>
          <div style='font-size:22px; color:{"green" if label=="Positive" else "orange" if label=="Neutral" else "red"};'>
            {emoji_dict[label]} <b>{label}</b>
            <span style='font-size:16px;'>(Confidence: {conf:.2f}%)</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # — GAP before breakdown
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        # ── Confidence Breakdown ─────────────────────────────────────
        with col1:
            with st.container():
                st.markdown("""
                <div style='border:1px solid #ddd; border-radius:10px; padding:20px; width:100%;'>
                  <h4 style='text-align:center;'>📈 Confidence Breakdown</h4>
                """, unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(2.5,2))
                labels = ["Positive","Neutral","Negative"]
                vals = [probs[cls.index("Positive")], probs[cls.index("Neutral")], probs[cls.index("Negative")]]
                colors = ['#28a745','#ffc107','#dc3545']
                ax.pie(vals, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
                st.markdown("</div>", unsafe_allow_html=True)

        # ── Review Analysis ─────────────────────────────────────────
        with col2:
            with st.container():
                st.markdown("""
                <div style='border:1px solid #ddd; border-radius:10px; padding:20px; width:100%;'>
                  <h4 style='text-align:center;'>📊 Review Analysis</h4>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                  <div style='padding:12px;'>
                    <ul style='font-size:16px;'>
                      <li><b>📝 Length:</b> {L} characters</li>
                      <li><b>📚 Words:</b> {W}</li>
                      <li><b>❗❗ Exclamations:</b> {E}</li>
                      <li><b>😃 Emojis:</b> {emo_ct}</li>
                      <li><b>❤️ Sentiment Score:</b> {score:.3f}</li>
                    </ul>
                  </div>
                """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

        # ─── Download & Footer ───────────────────────────────────────
        df = pd.DataFrame([{
            "Review": user_input,
            "Prediction": label,
            "Confidence": f"{conf:.2f}%",
            "Length": L,
            "Words": W,
            "Exclamations": E,
            "Emojis": emo_ct,
            "Sentiment Score": score
        }])
        dl1, dl2, dl3 = st.columns([2,6,2])
        with dl2:
            st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name="result.csv", use_container_width=True)
            st.markdown("""
            <div style='text-align:center; padding-top:10px;'>
              <span style='font-size:13px; color:gray;'>🤖 Powered by Neural Network (MLP) | TF-IDF + Extras</span>
            </div>
            """, unsafe_allow_html=True)
