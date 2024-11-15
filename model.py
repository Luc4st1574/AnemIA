import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class AnemiaModel:
    def __init__(self, data_path="AnemIA\\DataSet\\Anemia_Dataset.csv", model_path="AnemIA\\Model\\anemia_model.pkl"):
        self.data_path = data_path
        self.model_path = model_path
        self.model = None

    def load_data(self):
        """Load and preprocess the dataset."""
        data = pd.read_csv(self.data_path)
        data = data.drop('Name', axis=1)
        data['Sex'] = data['Sex'].replace({'M': 0, 'F': 1})
        data['Anemic'] = data['Anemic'].replace({'No': 0, 'Yes': 1})
        return data

    def preprocess_data(self, data):
        """Split the data."""
        X = data[['Sex', 'R', 'G', 'B']]
        y = data[['Hb', 'Anemic']]  # Multi-output: Hb (regression) and Anemic (classification)
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        """Train the model for multi-output prediction and save the best one."""
        data = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(data)

        # Define the regressor for multi-output
        regressor = RandomForestRegressor(random_state=42)
        multi_output_regressor = MultiOutputRegressor(regressor)

        # Perform GridSearchCV for hyperparameter tuning
        param_grid = {
            'estimator__n_estimators': [100, 200],
            'estimator__max_depth': [None, 10]
        }
        grid_search = GridSearchCV(multi_output_regressor, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='r2')
        grid_search.fit(X_train, y_train)

        # Best model
        self.model = grid_search.best_estimator_
        print(f"Best model parameters: {grid_search.best_params_}")

        # Save and evaluate the best model
        self.save_model()
        self.evaluate(X_test, y_test)

    def evaluate(self, X_test, y_test):
        """Evaluate the trained model."""
        y_pred = self.model.predict(X_test)

        # Separate Hb and Anemic outputs
        y_test_hb, y_test_anemic = y_test['Hb'], y_test['Anemic']
        y_pred_hb, y_pred_anemic = y_pred[:, 0], (y_pred[:, 1] > 0.5).astype(int)

        # Metrics for Hb (regression)
        mse = mean_squared_error(y_test_hb, y_pred_hb)
        mae = mean_absolute_error(y_test_hb, y_pred_hb)
        r2 = r2_score(y_test_hb, y_pred_hb)
        print("Regression - Hb Levels:")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"R² Score: {r2:.2f}")

        # Metrics for Anemic (classification)
        accuracy = accuracy_score(y_test_anemic, y_pred_anemic)
        print("Classification - Anemia:")
        print(f"Accuracy: {accuracy:.2f}")
        print(classification_report(y_test_anemic, y_pred_anemic))

        # Plot regression results
        try:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=y_test_hb, y=y_pred_hb)
            plt.plot([y_test_hb.min(), y_test_hb.max()], [y_test_hb.min(), y_test_hb.max()], 'k--', lw=2)
            plt.xlabel('Actual Hb Levels')
            plt.ylabel('Predicted Hb Levels')
            plt.title('Actual vs Predicted Hb Levels')
            plt.tight_layout()
            regression_path = 'AnemIA\\Model\\regression_evaluation.png'
            os.makedirs(os.path.dirname(regression_path), exist_ok=True)
            plt.savefig(regression_path)
            plt.close()
            print(f"Regression evaluation plot saved at: {regression_path}")
        except Exception as e:
            print(f"Error generating regression plot: {e}")

        # Plot confusion matrix for classification
        try:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test_anemic, y_pred_anemic)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Anemia', 'Anemia'], yticklabels=['No Anemia', 'Anemia'])
            plt.title('Confusion Matrix - Anemia')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            confusion_matrix_path = 'AnemIA\\Model\\confusion_matrix.png'
            os.makedirs(os.path.dirname(confusion_matrix_path), exist_ok=True)
            plt.savefig(confusion_matrix_path)
            plt.close()
            print(f"Confusion matrix plot saved at: {confusion_matrix_path}")
        except Exception as e:
            print(f"Error generating confusion matrix plot: {e}")


    def save_model(self):
        """Save the trained model."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Please train the model first.")
        self.model = joblib.load(self.model_path)
        print(f"Model loaded: {type(self.model)}")

    def predict(self, input_data):
        """Make predictions using the trained model."""
        if self.model is None:
            self.load_model()

        # Predict Hb levels and Anemia status
        prediction = self.model.predict(input_data)
        hb_prediction = prediction[:, 0]
        anemia_prediction = (prediction[:, 1] > 0.5).astype(int)

        # Create a dictionary with feature names and their values
        features = dict(zip(['Sex', 'R', 'G', 'B'], input_data[0]))

        # Return predictions and details
        details = {
            'predicted_Hb': hb_prediction[0],
            'predicted_Anemic': 'Anemia' if anemia_prediction[0] == 1 else 'No Anemia',
            'features': features
        }

        return details
