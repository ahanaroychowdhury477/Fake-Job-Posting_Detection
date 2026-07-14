from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import PyPDF2

app = Flask(__name__)

# Load model
model = joblib.load("job_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    job = request.form.get("job", "")

    uploaded_file = request.files.get("file")

    if uploaded_file:

        if uploaded_file.filename.endswith(".txt"):
            job = uploaded_file.read().decode("utf-8")

        elif uploaded_file.filename.endswith(".pdf"):

            reader = PyPDF2.PdfReader(uploaded_file)

            text = ""

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text

            job = text

    if job.strip() == "":
        return jsonify({
            "status": "warning",
            "message": "Please enter a job description."
        })

    job_vector = vectorizer.transform([job])

    prediction = model.predict(job_vector)[0]

    probability = model.predict_proba(job_vector)

    confidence = round(np.max(probability) * 100, 2)

    if prediction == 1:
        result = "Fake Job Posting"
    else:
        result = "Real Job Posting"

    return jsonify({
        "status": "success",
        "prediction": result,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)