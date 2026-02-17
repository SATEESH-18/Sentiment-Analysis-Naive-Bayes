import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data (only first run)
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')   # ← ADD THIS
nltk.download('stopwords')

# Load saved model and vectorizer
model, vectorizer = pickle.load(open("sentiment_model.pkl", "rb"))

# Stopwords
stop_words = set(stopwords.words("english"))

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.title("🍽 Restaurant Review Sentiment Analysis")
st.write("Enter a review below to predict whether it is Positive or Negative.")

user_input = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]

        if prediction == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")