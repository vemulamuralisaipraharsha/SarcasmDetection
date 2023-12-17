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

classifier = joblib.load('trained_model.joblib')

import json

# Read data from JSON file
with open('output.json', 'r') as file:
    data = json.load(file)

# Access the sentences array
test_texts = data['sentences']

# Print the sentences
# for sentence in sentences:
#     print(sentence)

# Test with custom inputs
# test_texts = [
#     "Well, that's just fantastic! I love getting stuck in traffic.",  # Not sarcastic
#     "Brilliant idea, let's have a meeting to decide when to schedule the next meeting.",  # Sarcastic
#     "Sure, I'd love to work late on a Friday night. What could be more fun?",  # Not sarcastic
#     "Oh, great! Another flat tire on my way to work.",  # Sarcastic
# ]
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

url = "./Sarcasm_Dataset.json"
df = pd.read_json(url, lines=True)

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha()]  # Lemmatize and convert to lowercase
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

df['headline'] = df['headline'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(df['headline'], df['is_sarcastic'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Preprocess test texts
test_texts_preprocessed = [preprocess_text(text) for text in test_texts]

# Vectorize test texts
test_texts_vectorized = vectorizer.transform(test_texts_preprocessed)

# Make predictions for test texts
test_predictions = classifier.predict(test_texts_vectorized)

# Display results for test inputs
# for text, prediction in zip(test_texts, test_predictions):
#     print(f"\nText: {text}")
#     if prediction == 0:
#         print("Prediction: Sarcastic")
#     else:
#         print("Prediction: Not Sarcastic")
        
results = []

# Your loop for generating text and prediction
for text, prediction in zip(test_texts, test_predictions):
    result = {'text': text, 'prediction': 'Sarcastic' if prediction == 1 else 'Not Sarcastic'}
    results.append(result)

# Save the results to a JSON file
with open('predicted.json', 'w') as file:
    json.dump(results, file, indent=2)

print("Results saved to predicted.json")