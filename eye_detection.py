import cv2
import numpy as np

class EyeRGBDetector:
    def __init__(self, webcam_index=0):
        self.cap = cv2.VideoCapture(webcam_index)
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def get_frame_with_eyes(self):
        frame = self.capture_frame()
        if frame is None:
            return None, []

        frame = self.color_correct(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        eyes = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            
            eyes_in_face = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(w//3, h//3))
            
            for (ex, ey, ew, eh) in eyes_in_face:
                eyes.append((x+ex, y+ey, ew, eh))

        return frame, eyes

    def capture_eye_rgb_auto(self):
        frame, eyes = self.get_frame_with_eyes()

        if len(eyes) == 0:
            return None

        (x, y, w, h) = eyes[0]
        eye_region = frame[y:y+h, x:x+w]

        height, width = eye_region.shape[:2]
        center_y, center_x = height // 2, width // 2
        roi = eye_region[center_y-height//4:center_y+height//4, center_x-width//4:center_x+width//4]
        
        avg_color = cv2.mean(roi)[:3]

        return avg_color

    def color_correct(self, frame):
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        
        # Increase red (decrease blue)
        a = cv2.add(a, 10)
        # Increase yellow (decrease blue)
        b = cv2.add(b, 15)
        
        # Merge channels
        limg = cv2.merge((cl, a, b))
        
        # Convert back to BGR color space
        corrected = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Adjust color balance
        corrected = cv2.addWeighted(corrected, 1.2, np.zeros(corrected.shape, corrected.dtype), 0, -30)
        
        return corrected

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

