import streamlit as st
import joblib
import numpy as np
import PyPDF2


st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🛡️",
    layout="wide"
)


model = joblib.load("job_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


st.sidebar.title("🛡️ Fake Job Detector")
st.sidebar.markdown("---")

st.sidebar.subheader("Model")
st.sidebar.write("Algorithm: Logistic Regression")
st.sidebar.write("Vectorizer: TF-IDF")

st.sidebar.markdown("---")

st.sidebar.subheader("Performance")
st.sidebar.metric("Accuracy", "96.7%")


st.title("🛡️ Fake Job Posting Detection")

st.write(
    "Detect fraudulent job postings using Machine Learning and Natural Language Processing."
)

uploaded_file = st.file_uploader(
    "📂 Upload Job Description (TXT/PDF)",
    type=["txt", "pdf"]
)

job = ""

if uploaded_file is not None:

    if uploaded_file.type == "text/plain":
        job = uploaded_file.read().decode("utf-8")

    elif uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                job += text

job = st.text_area(
    "Enter Job Description",
    key="job",
    height=180,
    placeholder="Paste the complete job description here..."
)
if job:
    st.caption(f"📝 Words: {len(job.split())} | 🔤 Characters: {len(job)}")
if st.button("🗑️ Clear"):
    st.rerun()


if st.button("🔍 Predict", use_container_width=True):
    

    if job.strip() == "":
        st.warning("⚠ Please enter a job description.")
    else:

        
        job_vector = vectorizer.transform([job])

        
        prediction = model.predict(job_vector)

        
        probability = model.predict_proba(job_vector)
        confidence = np.max(probability) * 100

        st.markdown("---")
        st.subheader("📋 Prediction Result")

        if prediction[0] == 1:
            st.error("⚠ Fake Job Posting")
            st.write(
                "Reason: The model detected suspicious patterns commonly found in fraudulent job advertisements."
            )
        else:
            st.success("✅ Real Job Posting")
            st.write(
                "Reason: The posting appears to match patterns commonly found in genuine job advertisements."
            )

        st.metric("Confidence",f"{confidence:.2f}%")
        st.progress(confidence / 100)
        st.markdown("---")
st.caption("© 2026 Fake Job Posting Detection | Machine Learning Project")