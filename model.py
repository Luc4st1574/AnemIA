import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class AnemiaModel:
    def __init__(self, data_path="AnemIA\\DataSet\\Anemia_Dataset.csv", model_path="AnemIA\\Model\\anemia_model.pkl"):
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
        """Split the data."""
        X = data[['Sex', 'R', 'G', 'B', 'Hb']]
        y = data['Anemic']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        """Train the Random Forest model with hyperparameter tuning and save it."""
        data = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(data)

        # Create a pipeline with scaling and Random Forest
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state=42))
        ])

        # Define hyperparameters to tune
        param_grid = {
            'rf__n_estimators': [100, 200, 300],
            'rf__max_depth': [None, 5, 10, 15],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4]
        }

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        # Get the best model
        self.model = grid_search.best_estimator_

        # Save the model
        self.save_model()

        # Evaluate the model
        self.evaluate(X_test, y_test)

        # Analyze feature importance
        self.analyze_feature_importance(X_train.columns)

    def evaluate(self, X_test, y_test):
        """Evaluate the trained model."""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')
        print(classification_report(y_test, y_pred))

        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('AnemIA\\Model\\confusion_matrix.png')
        plt.close()

    def analyze_feature_importance(self, feature_names):
        """Analyze and plot feature importance."""
        importances = self.model.named_steps['rf'].feature_importances_
        feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_imp = feature_imp.sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_imp)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('AnemIA\\Model\\feature_importance.png')
        plt.close()

    def save_model(self):
        """Save the trained model and scaler."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load the trained model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Please train the model first.")
        self.model = joblib.load(self.model_path)

    def predict(self, input_data):
        """Make a prediction using the trained model."""
        if self.model is None:
            self.load_model()
        return self.model.predict(input_data)