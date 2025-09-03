import streamlit as st
import joblib
import docx2txt

# Load the trained Decision Tree pipeline
model = joblib.load("decision_tree_pipeline.pkl")

st.set_page_config(page_title="Resume Classification App", layout="centered")

st.title(" Resume Classification App ")

st.write("Upload a resume file (.txt or .docx) OR paste text to classify into one of: "
         "**Peoplesoft Resume, React Developer, SQL Developer, Workday**.")

# --- Option 1: File Upload ---
uploaded_file = st.file_uploader(" Upload a resume file", type=["txt", "docx"])

resume_text = ""

if uploaded_file is not None:
    # If it's a .txt file
    if uploaded_file.type == "text/plain":
        resume_text = uploaded_file.read().decode("utf-8")

    # If it's a .docx file
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume_text = docx2txt.process(uploaded_file)

# --- Option 2: Manual Text Input ---
manual_text = st.text_area(" Or paste your resume text here:", height=200)

# Use uploaded text if available, otherwise use manual input
final_text = resume_text if resume_text.strip() != "" else manual_text

if st.button("Predict"):
    if final_text.strip():
        prediction = model.predict([final_text])[0]
        st.success(f" Predicted Category: **{prediction}**")
    else:
        st.warning("Please upload a file or paste some text before predicting.")
