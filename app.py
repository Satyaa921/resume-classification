import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

# App title
st.title("Resume Classification App (Random Forest)")

# User input
resume_text = st.text_area("Paste your resume text here:")

# Prediction button
if st.button("Predict"):
    if resume_text.strip() == "":
        st.warning("Please enter some resume text first.")
    else:
        # Convert input text using saved vectorizer
        X = vectorizer.transform([resume_text])
        
        # Predict using the Random Forest model
        prediction = model.predict(X)[0]
        
        # Show result
        st.success(f" Predicted Category: **{prediction}**")
