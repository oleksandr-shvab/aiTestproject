import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def numeric_analysis():
    print("\nNumeric Analysis")

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


if __name__ == "__main__":
    numeric_analysis()
