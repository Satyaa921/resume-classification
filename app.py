import streamlit as st
import joblib

# Load the saved Decision Tree pipeline
model = joblib.load("decision_tree_pipeline.pkl")  

st.set_page_config(page_title="Resume Classification App", layout="centered")

st.title(" Resume Classification App (Decision Tree)")

st.write("This app classifies resumes into one of four categories: "
         "**Peoplesoft Resume, React Developer, SQL Developer, Workday**.")

# Text input
resume_text = st.text_area("Paste your resume text here:", height=200)

if st.button("Predict"):
    if resume_text.strip():
        prediction = model.predict([resume_text])[0]
        st.success(f" Predicted Category: **{prediction}**")
    else:
        st.warning(" Please paste some resume text before predicting.")
