# model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

class AnemiaModel:
    def __init__(self, data_path="AnemIA\\Anemia_Dataset.csv", model_path="AnemIA\\anemia_model.pkl"):
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.scaler = None

    def load_data(self):
        """Load and preprocess the dataset."""
        data = pd.read_csv(self.data_path)
        data = data.drop('Name', axis=1)
        data['Sex'] = data['Sex'].replace({'M': 0, 'F': 1})
        data['Anemic'] = data['Anemic'].replace({'No': 0, 'Yes': 1})
        return data

    def preprocess_data(self, data):
        """Split and scale the data."""
        X = data[['Sex', 'R', 'G', 'B', 'Hb']]
        y = data['Anemic']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train(self):
        """Train the Logistic Regression model and save it."""
        data = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(data)
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)
        self.save_model()
        self.evaluate(X_test, y_test)

    def evaluate(self, X_test, y_test):
        """Evaluate the trained model."""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')
        print(classification_report(y_test, y_pred))

    def save_model(self):
        """Save the trained model and scaler."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load the trained model and scaler."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Please train the model first.")
        model_data = joblib.load(self.model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']

    def predict(self, input_data):
        """Make a prediction using the trained model."""
        if self.model is None or self.scaler is None:
            self.load_model()
        input_scaled = self.scaler.transform(input_data)
        prediction = self.model.predict(input_scaled)
        return prediction