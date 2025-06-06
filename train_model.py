import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv('UpdatedResumeDataSet.csv')  # Change to your file path

# 2. Clean text function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)       # Remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)    # Remove non-alphabet chars
    text = text.lower()                        # Lowercase
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

# Apply cleaning
df['cleaned_resume'] = df['Resume'].apply(clean_text)

# 3. Feature extraction
tfidf = TfidfVectorizer(stop_words='english', max_features=1500)
X = tfidf.fit_transform(df['cleaned_resume']).toarray()
y = df['Category']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 7. Save model and vectorizer
with open('resume_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Model and vectorizer saved successfully!")
