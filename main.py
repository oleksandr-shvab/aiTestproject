import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the model and vectorizer
model = joblib.load('sentiment_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')


def predict_sentiment(text):
    # Vectorize the input text
    vectorized_text = tfidf.transform([text])

    # Make a prediction
    prediction = model.predict(vectorized_text)

    return prediction[0]


def main():
    while True:
        # Get user input
        user_input = input("Enter text to analyze sentiment, or type 'exit' to quit: ")

        if user_input.lower() == 'exit':
            break

        # Predict sentiment
        sentiment = predict_sentiment(user_input)
        print(f"Predicted Sentiment: {sentiment}")

    # # Train model
    # train_file_path = '/home/alexandr/Downloads/train.csv'
    # test_file_path = '/home/alexandr/Downloads/test.csv'
    #
    # train_data = pd.read_csv(train_file_path, encoding='ISO-8859-1')
    # test_data = pd.read_csv(test_file_path, encoding='ISO-8859-1')
    #
    # train_data['text'] = train_data['text'].fillna('')
    # test_data['text'] = test_data['text'].fillna('')
    #
    # train_data['sentiment'] = train_data['sentiment'].astype(str)
    # test_data['sentiment'] = test_data['sentiment'].astype(str)
    #
    # tfidf = TfidfVectorizer(max_features=5000)  # Limiting to 5000 features
    # X_train = tfidf.fit_transform(train_data['text'])
    # y_train = train_data['sentiment']
    #
    # # Model Training
    # model = LogisticRegression(max_iter=1000)
    # model.fit(X_train, y_train)
    #
    # # Applying the same transformation to the test data
    # X_test = tfidf.transform(test_data['text'])
    # y_test = test_data['sentiment']
    #
    # # Model Evaluation
    # predictions = model.predict(X_test)
    # report = classification_report(y_test, predictions)
    # print(report)
    #
    # # Save the model
    # joblib.dump(model, 'sentiment_model.pkl')
    #
    # # Save the TF-IDF vectorizer
    # joblib.dump(tfidf, 'tfidf_vectorizer.pkl')


if __name__ == "__main__":
    main()
