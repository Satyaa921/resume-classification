import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("decision_tree_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

st.title("Resume Classification App (Decision Tree)")
st.write("This app classifies resumes into categories using a Decision Tree model.")

# User input
resume_text = st.text_area("Paste your resume text here:")

if st.button("Predict"):
    if resume_text.strip() != "":
        # Transform input text
        X = vectorizer.transform([resume_text])
        
        # Predict
        prediction = model.predict(X)[0]
        
        st.success(f"Predicted Category: **{prediction}**")
    else:
        st.warning(" Please paste some text before predicting.")
