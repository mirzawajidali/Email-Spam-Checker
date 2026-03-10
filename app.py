"""
Streamlit UI for the BERT Spam Email Classifier.
Run with: streamlit run app.py
"""

import streamlit as st
from predict import SpamPredictor


@st.cache_resource
def load_model():
    """Load model once and cache it across reruns."""
    return SpamPredictor(model_dir="saved_model")


st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="centered")

st.title("Spam Email Detector")
st.write("Paste an email below to check if it's **spam** or **not spam**.")

email_text = st.text_area("Email Content", height=200, placeholder="Paste your email content here...")

if st.button("Check", type="primary", use_container_width=True):
    if not email_text.strip():
        st.warning("Please paste some email content first.")
    else:
        with st.spinner("Analyzing..."):
            predictor = load_model()
            result = predictor.predict(email_text)

        label = result["label"]
        confidence = result["confidence"]
        ham_prob = result["probabilities"]["ham"]
        spam_prob = result["probabilities"]["spam"]

        if label == "spam":
            st.error(f"🚫 **SPAM** — Confidence: {confidence:.1%}")
        else:
            st.success(f"✅ **Not Spam** — Confidence: {confidence:.1%}")

        col1, col2 = st.columns(2)
        col1.metric("Not Spam", f"{ham_prob:.1%}")
        col2.metric("Spam", f"{spam_prob:.1%}")

        st.progress(spam_prob, text="Spam probability")
