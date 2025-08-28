import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack

# Load model and vectorizer only
model = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

# Function to add extra features with simple scaling
def add_extra_features(texts, X_tfidf):
    word_count = np.array([len(s.split()) for s in texts]).reshape(-1,1)
    char_count = np.array([len(s) for s in texts]).reshape(-1,1)
    avg_wordlen = (char_count / (word_count + 1)).reshape(-1,1)
    
    feats = np.hstack([word_count, char_count, avg_wordlen])
    # Simple normalization (z-score scaling)
    feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-6)
    
    return hstack([X_tfidf, feats])

# Streamlit app
st.title("Resume Classification App (Random Forest)")

resume_text = st.text_area("Paste your resume text here:")

if st.button("Predict"):
    if resume_text.strip() == "":
        st.warning(" Please enter some resume text.")
    else:
        try:
            # TF-IDF transformation
            X_tfidf = vectorizer.transform([resume_text])
            # Add extra features with scaling
            X_final = add_extra_features([resume_text], X_tfidf)
            # Prediction
            prediction = model.predict(X_final)[0]
            st.success(f"Predicted Category: **{prediction}**")
        except Exception as e:
            st.error(f" Error during prediction: {e}")
