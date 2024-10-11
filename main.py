# main.py
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

    # Initialize the eye detector
    eye_detector = EyeRGBDetector()  # Try different indices (0, 1, 2)

    # Initialize and run the UI
    app = ControlPanel(eye_detector, model)
    app.mainloop()

if __name__ == "__main__":
    main()
