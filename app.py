import streamlit as st
import joblib
import numpy as np

# Load the trained pipeline
model = joblib.load("final_resume_model.pkl")

# App UI
st.title(" Resume Classification App")
st.write("This app classifies resumes into categories using a Random Forest pipeline trained on balanced data.")

# User input
resume_text = st.text_area("Paste your resume text here:")

if st.button("Predict"):
    if resume_text.strip() == "":
        st.warning(" Please enter some text before predicting.")
    else:
        # Get prediction
        prediction = model.predict([resume_text])[0]

        # Get prediction probabilities
        probs = model.predict_proba([resume_text])[0]
        class_labels = model.classes_

        # Show main result
        st.success(f" Predicted Category: **{prediction}**")

        # Show probabilities per class
        st.subheader("Prediction Probabilities:")
        for label, p in zip(class_labels, probs):
            st.write(f"- {label}: {p*100:.2f}%")
