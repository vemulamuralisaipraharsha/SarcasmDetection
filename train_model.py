# Make sure you have pandas, nltk, scikit-learn installed
# You can install them using: pip install pandas nltk scikit-learn

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import joblib

import nltk
nltk.download('wordnet')

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

print("[LOG] Data Acquiring...")

# Load the dataset
# url = "https://raw.githubusercontent.com/vemulapraharsha/SarcasmDetection/main/Sarcasm_Dataset.json"
url = "./Sarcasm_Dataset.json"
df = pd.read_json(url, lines=True)

print("[LOG] Data Acquired")

# Preprocess the data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha()]  # Lemmatize and convert to lowercase
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

df['headline'] = df['headline'].apply(preprocess_text)

# Split data into training and 
# testing sets
X_train, X_test, y_train, y_test = train_test_split(df['headline'], df['is_sarcastic'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train an SVM classifier
classifier = SVC(kernel='linear', C=1)
classifier.fit(X_train_vectorized, y_train)

joblib.dump(classifier, 'trained_model.joblib')


# # Make predictions
# predictions = classifier.predict(X_test_vectorized)

# print("[LOG] Evaluating...")

# # Evaluate the model
# accuracy = accuracy_score(y_test, predictions)
# print(f"Accuracy: {accuracy}")
# print("\nClassification Report:\n", classification_report(y_test, predictions))