import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("emails.csv")

X = data["text"].fillna("")
y = data["spam"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nSPAM MAIL PREDICTION SYSTEM")

while True:
    email = input("\nEnter email text (type 'exit' to stop): ")

    if email.lower() == "exit":
        break

    email_vector = vectorizer.transform([email])
    prediction = model.predict(email_vector)[0]
    probability = model.predict_proba(email_vector).max()

    if prediction == 1:
        print("Prediction: SPAM MAIL")
    else:
        print("Prediction: NOT SPAM")

    print("Confidence:", round(probability * 100, 2), "%")