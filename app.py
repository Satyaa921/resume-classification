import streamlit as st
import joblib

# Load the trained Random Forest model and TF-IDF vectorizer
model = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

# App title
st.title("Resume Classification App (Random Forest)")

# Instructions
st.markdown(
    "This app classifies resumes into categories using a Random Forest model "
    "trained on resume text. Paste your resume text below to get a prediction."
)

# Text input
resume_text = st.text_area("Paste your resume text here:")

# Prediction
if st.button("Predict"):
    if resume_text.strip() == "":
        st.warning("Please enter some resume text.")
    else:
        try:
            # Transform input text using the saved vectorizer
            X = vectorizer.transform([resume_text])

            # Predict using the trained model
            prediction = model.predict(X)[0]

            # Show result
            st.success(f" Predicted Category: **{prediction}**")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
