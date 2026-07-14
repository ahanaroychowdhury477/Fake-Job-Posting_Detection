import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv("fake_job_postings.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.isnull().sum())
print(df["fraudulent"].unique())
df["text"] = (
df["title"].fillna("") + " " +
df["company_profile"].fillna("") + " " +
df["description"].fillna("") + " " +
df["requirements"].fillna("") + " " +
df["benefits"].fillna("") + " " +
df["industry"].fillna("") + " " +
df["employment_type"].fillna("")
)

print(df["text"].head())
X = df["text"]
y = df["fraudulent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", len(X_train))
print("Testing Data:", len(X_test))

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1,2)
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print(X_train.shape)
print(X_test.shape)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=2,
    solver="liblinear"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
job = input("Enter Job Description: ")

job_vector = vectorizer.transform([job])

prediction = model.predict(job_vector)

if prediction[0] == 1:
    print("⚠️ Fake Job Posting")
else:
    print("✅ Real Job Posting")
joblib.dump(model, "job_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model Saved Successfully!")   