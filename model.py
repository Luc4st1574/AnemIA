import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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
        self.feature_importance = None

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
        """Train the model using different classifiers with hyperparameter tuning and save the best one."""
        data = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(data)

        # Create a pipeline with scaling
        pipeline = Pipeline([('scaler', StandardScaler())])

        # Define the classifiers to evaluate
        classifiers = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'KNeighbors': KNeighborsClassifier(),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
        }

        # Define hyperparameters for each classifier
        param_grids = {
            'RandomForest': {
                'randomforest__n_estimators': [100, 200],
                'randomforest__max_depth': [None, 10]
            },
            'GradientBoosting': {
                'gradientboosting__n_estimators': [100, 200],
                'gradientboosting__learning_rate': [0.01, 0.1],
                'gradientboosting__max_depth': [3, 5]
            },
            'SVM': {
                'svm__C': [0.1, 1, 10],
                'svm__kernel': ['linear', 'rbf']
            },
            'KNeighbors': {
                'kneighbors__n_neighbors': [3, 5, 7],
                'kneighbors__weights': ['uniform', 'distance']
            },
            'LogisticRegression': {
                'logisticregression__C': [0.1, 1, 10]
            }
        }

        best_model = None
        best_score = 0

        # Iterate over classifiers and perform GridSearchCV
        for name, classifier in classifiers.items():
            pipeline.steps.append((name.lower(), classifier))

            param_grid = param_grids.get(name)
            if param_grid:
                grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
                grid_search.fit(X_train, y_train)

                score = grid_search.best_score_
                print(f"{name} best score: {score:.2f}")

                if score > best_score:
                    best_score = score
                    best_model = grid_search.best_estimator_

            # Remove classifier from pipeline for next iteration
            pipeline.steps.pop()

        # Set the best model
        self.model = best_model

        # After training, store the feature importance if available
        if 'randomforest' in self.model.named_steps:
            self.feature_importance = self.model.named_steps['randomforest'].feature_importances_

        # Save and evaluate the best model
        self.save_model()
        self.evaluate(X_test, y_test)


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
        if self.feature_importance is not None:
            importances = self.feature_importance
            feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_imp = feature_imp.sort_values('importance', ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_imp)
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig('AnemIA\\Model\\feature_importance.png')
            plt.close()

    def save_model(self):
        """Save the trained model and feature importance."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_importance': self.feature_importance
        }, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load the trained model and feature importance."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Please train the model first.")
        model_data = joblib.load(self.model_path)
        self.model = model_data['model']
        self.feature_importance = model_data['feature_importance']

    def predict(self, input_data):
        """Make a prediction using the trained model and return detailed information."""
        if self.model is None:
            self.load_model()
        
        # Make prediction
        prediction = self.model.predict(input_data)[0]
        probability = self.model.predict_proba(input_data)[0]

        # Get feature importance
        if self.feature_importance is None and 'rf' in self.model.named_steps:
            self.feature_importance = self.model.named_steps['rf'].feature_importances_

        # Create a dictionary with feature names and their values
        features = dict(zip(['Sex', 'R', 'G', 'B', 'Hb'], input_data[0]))

        # Create detailed information
        details = {
            'prediction': 'Anemia' if prediction == 1 else 'No Anemia',
            'probability': probability[1] if prediction == 1 else probability[0],
            'features': features,
            'feature_importance': dict(zip(['Sex', 'R', 'G', 'B', 'Hb'], self.feature_importance))
        }

        return details
