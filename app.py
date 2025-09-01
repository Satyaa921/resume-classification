import streamlit as st
import joblib

# Load the saved pipeline (vectorizer + Decision Tree model)
pipeline = joblib.load("decision_tree_pipeline.pkl")

# App title and description
st.title("Resume Classification App (Decision Tree)")
st.write("This app classifies resumes into categories using a Decision Tree model trained on resume text.")

# Text area for user input
resume_text = st.text_area("Paste your resume text here:")

# Predict button
if st.button("Predict"):
    if resume_text.strip() != "":
        # Pipeline handles preprocessing + prediction
        prediction = pipeline.predict([resume_text])[0]
        
        st.success(f"Predicted Category: **{prediction}**")
    else:
        st.warning(" Please paste some text before predicting.")
