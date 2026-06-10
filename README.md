
# Emotion Classifier Streamlit App

This repository contains the code for an Emotion Classifier Streamlit web application. The app allows users to input text and get a prediction of the emotion expressed in the text. It leverages several machine learning models (SVM, RandomForest, Logistic Regression) trained on text data and uses an ensemble method for improved accuracy.

## Project Structure

- `app.py`: The Streamlit application script.
- `requirements.txt`: Lists the Python dependencies required to run the app.
- `.joblib` files: Saved machine learning models (SVM, RandomForest, Logistic Regression, Ensemble) and the `TfidfVectorizer` used for text preprocessing.

## Models Used

The following models were trained and evaluated:
- **Multinomial Naive Bayes (Bag of Words)**
- **Multinomial Naive Bayes (TF-IDF)**
- **Logistic Regression (TF-IDF)**
- **K-Nearest Neighbors (TF-IDF)**
- **Support Vector Machine (TF-IDF)**
- **RandomForest Classifier (TF-IDF)**

Hyperparameter tuning was performed using `GridSearchCV` for SVM, RandomForest, and Logistic Regression models. An ensemble `VotingClassifier` combining the best versions of these three models was also implemented.

## Best Performing Models (after GridSearchCV)

- **Logistic Regression Model (TF-IDF)**: Achieved the highest test accuracy of 0.8978.
- **Ensemble Model (Soft Voting)**: Achieved a test accuracy of 0.8919.
- **SVM Model (TF-IDF)**: Achieved a test accuracy of 0.8831.
- **RandomForest Model (TF-IDF)**: Achieved a test accuracy of 0.8819.

## How to Run the Streamlit App (Locally or on Colab)

### Prerequisites

- Python 3.x
- `pip` package manager
- Git
- Access to Google Drive (if running on Colab and models are stored there initially)

### 1. Clone the Repository

```bash
git clone https://github.com/Ashilshaikh/Emotion-detection.git # Replace with your repo URL
cd Emotion-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Ensure Models and Vectorizer are Present

The `.joblib` model files and `TfidfVectorizer.joblib` are crucial for the app to function. They should be present in the same directory as `app.py`. If you've pushed them using Git LFS, they will be downloaded automatically when you clone the repository. If you are running this in Google Colab, the notebook takes care of copying them from Google Drive to the local environment where `app.py` runs.

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

This command will open the Streamlit application in your web browser. If running on Google Colab, you might need to use `npx localtunnel` to expose the port (as shown in the Colab notebook).

## Text Preprocessing Steps

Before feeding text to the models, the following preprocessing steps are applied:
- Convert text to lowercase.
- Remove punctuation.
- Remove numbers.
- Remove emojis/non-ASCII characters.
- Remove stopwords and tokenize the text.
- Transform text using the pre-trained `TfidfVectorizer`.

---

**Note**: The NLTK data (`stopwords`, `punkt`) is downloaded programmatically within `app.py` if not already present.

