import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


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
    # Load your dataset
    data = pd.read_csv('cirrhosis.csv')

    # Handling missing values
    data.fillna(method='ffill', inplace=True)

    # Separate features and target variable
    X = data.drop('Stage', axis=1)
    y = data['Stage']

    # Categorical and numerical column names
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Create a ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])

    # Create preprocessing and training pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', LogisticRegression())])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train the model
    model = pipeline.fit(X_train, y_train)

    joblib.dump(model, "cirrhosis_model.pkl")

    y_pred = model.predict(X_test)

    print(y_pred)

    # Print the classification report
    print(classification_report(y_test, y_pred))


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
