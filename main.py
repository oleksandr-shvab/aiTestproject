import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def make_prediction():
    if not os.path.exists('sentiment_model.pkl') or not os.path.exists('tfidf_vectorizer.pkl'):
        print("Model not found. Please train the model first.")
        return

    model = joblib.load('sentiment_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

    while True:
        user_input = input("Enter text to analyze sentiment: ")
        if user_input == 'exit':
            break

        vectorized_user_input = tfidf_vectorizer.transform([user_input])

        prediction = model.predict(vectorized_user_input)

        print(f"Predicted Sentiment: {prediction[0]}")


def train_model():
    """
    Train Logistic Regression model and save it
    """
    train_file_path = 'train.csv'

    train_data = pd.read_csv(train_file_path, encoding='ISO-8859-1')

    # Data Preprocessing
    train_data['text'] = train_data['text'].fillna('')
    train_data['sentiment'] = train_data['sentiment'].astype(str)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train = tfidf_vectorizer.fit_transform(train_data['text'])
    y_train = train_data['sentiment']

    # Model Training
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')


def main_menu():
    while True:
        print("\nSentiment Analysis")

        print("1. Train Model")
        print("2. Make a Prediction")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3):")

        if choice == '1':
            train_model()
        elif choice == '2':
            make_prediction()
        elif choice == '3':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main_menu()
