import joblib
import numpy as np
from scipy.sparse import hstack

# Load objects
model = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("tfidf.pkl")
scaler = joblib.load("scaler.pkl")

def add_extra_features(texts, X_tfidf):
    word_count = np.array([len(s.split()) for s in texts]).reshape(-1,1)
    char_count = np.array([len(s) for s in texts]).reshape(-1,1)
    avg_wordlen = (char_count / (word_count + 1)).reshape(-1,1)

    # Manually normalize (simple version)
    feats = np.hstack([word_count, char_count, avg_wordlen])
    feats = (feats - feats.mean(axis=0)) / feats.std(axis=0)  # z-score scaling

    return hstack([X_tfidf, feats])

# Streamlit app
st.title(" Resume Classification App (Random Forest)")

st.markdown(
    "This app classifies resumes into categories using a Random Forest model. "
    "Paste your resume text below and click Predict."
)

# User input
resume_text = st.text_area("Paste your resume text here:")

if st.button("Predict"):
    if resume_text.strip() == "":
        st.warning("Please enter some resume text.")
    else:
        try:
            # Step 1: TF-IDF transformation
            X_tfidf = vectorizer.transform([resume_text])

            # Step 2: Add the same extra features as in training
            X_final = add_extra_features([resume_text], X_tfidf)

            # Step 3: Prediction
            prediction = model.predict(X_final)[0]

            # Output
            st.success(f" Predicted Category: **{prediction}**")

        except Exception as e:
            st.error(f" Error during prediction: {e}")
