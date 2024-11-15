from model import AnemiaModel
from eye_detection import EyeRGBDetector
from ui import ControlPanel
import cv2

cv2.setUseOptimized(True)

def main():
    # Initialize the model
    model = AnemiaModel()
    try:
        model.load_model()
    except FileNotFoundError:
        print("Model not found. Training the model...")
        model.train()
        model.load_model()
        print("Model training complete. Visualizations saved in the Model directory.")

    # Initialize the eye detector
    eye_detector = EyeRGBDetector()  # Try different indices (0, 1, 2)

    # Initialize and run the UI
    app = ControlPanel(eye_detector, model)
    try:
        app.mainloop()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Application closed.")


if __name__ == "__main__":
    main()