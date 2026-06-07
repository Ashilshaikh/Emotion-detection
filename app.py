
import streamlit as st
import joblib
import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab') # Add this line to download punkt_tab
except LookupError:
    nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))

# --- Text Preprocessing Functions (replicate what was done during training) ---
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(txt):
    new = ""
    for i in txt:
        if not i.isdigit():
            new = new + i
    return new

def remove_emj(txt):
    new = ""
    for i in txt:
        if i.isascii():
            new += i
    return new

def remove_stopwords_and_tokenize(txt):
    words = word_tokenize(txt)
    cleaned = []
    for i in words:
        if i not in stop_words:
            cleaned.append(i)
    return ' '.join(cleaned)

def preprocess_text(text):
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_emj(text)
    text = remove_stopwords_and_tokenize(text)
    return text

# --- Load Models and Vectorizer ---
@st.cache_resource # Cache resource to avoid re-loading on every rerun
def load_resources():
    # The models are copied to the current directory before app.py is run.
    # So, we don't need to mount Google Drive or specify a Drive path here.
    model_save_dir = '.' # Load from current directory

    st.info(f"Loading models from: {model_save_dir}")
    loaded_ensemble_model = joblib.load(os.path.join(model_save_dir, 'ensemble_model.joblib'))
    st.info("Ensemble model loaded.")
    loaded_TfidfVectorizer = joblib.load(os.path.join(model_save_dir, 'TfidfVectorizer.joblib'))
    st.info("TfidfVectorizer loaded.")

    # Assuming unique_emotion was determined during training. Let's recreate it if not explicitly saved.
    emotion_labels = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']
    return loaded_ensemble_model, loaded_TfidfVectorizer, emotion_labels

ensemble_model, TfidfVectorizer, emotion_labels = load_resources()

# --- Streamlit App ---
st.title("Emotion Classifier")
st.write("Enter a sentence below to classify its emotion.")

user_input = st.text_area("Your Sentence:", "I am feeling very happy today!")

if st.button("Classify Emotion"):
    if user_input:
        # Preprocess input
        processed_input = preprocess_text(user_input)

        # Transform using the loaded TfidfVectorizer
        input_vectorized = TfidfVectorizer.transform([processed_input])

        # Predict emotion
        prediction = ensemble_model.predict(input_vectorized)

        # Map numerical prediction back to emotion label
        predicted_emotion = emotion_labels[prediction[0]]

        st.success(f"Predicted Emotion: **{predicted_emotion.capitalize()}**")
    else:
        st.warning("Please enter some text to classify.")
