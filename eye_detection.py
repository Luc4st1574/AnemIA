#eye_detection.py
import cv2

class EyeRGBDetector:
    def __init__(self, webcam_index=0):
        self.cap = cv2.VideoCapture(webcam_index)
        
        # Try to force camera to use default settings
        self.cap.set(cv2.CAP_PROP_SETTINGS, 1)
        
        # Load cascade classifiers
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def simple_white_balance(self, frame):
        # Split the frame into color channels
        b, g, r = cv2.split(frame)
        
        # Find the average of each channel
        r_avg = cv2.mean(r)[0]
        g_avg = cv2.mean(g)[0]
        b_avg = cv2.mean(b)[0]
        
        # Calculate scaling factors
        avg = (r_avg + g_avg + b_avg) / 3
        r_scale = avg / r_avg if r_avg > 0 else 1
        g_scale = avg / g_avg if g_avg > 0 else 1
        b_scale = avg / b_avg if b_avg > 0 else 1
        
        # Apply scaling to each channel
        r = cv2.multiply(r, r_scale)
        g = cv2.multiply(g, g_scale)
        b = cv2.multiply(b, b_scale)
        
        # Merge channels back
        balanced = cv2.merge([b, g, r])
        return balanced

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Apply white balance
        frame = self.simple_white_balance(frame)
        
        # Convert to YUV and back to BGR to normalize colors
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        return frame

    def get_frame_with_eyes(self):
        frame = self.capture_frame()
        if frame is None:
            return None, []

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(100, 100)
        )

        eyes = []
        for (x, y, w, h) in faces:
            # Only detect eyes in upper half of face
            face_gray = gray[y:y+h//2, x:x+w]
            
            eyes_in_face = self.eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.05,
                minNeighbors=7,
                minSize=(25, 25),
                maxSize=(w//3, h//4)
            )
            
            for (ex, ey, ew, eh) in eyes_in_face:
                eyes.append((x+ex, y+ey, ew, eh))
                # Draw rectangle around eyes
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

        # Draw diagnostic information
        cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Eyes detected: {len(eyes)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame, eyes

    def capture_eye_rgb_auto(self):
        frame, eyes = self.get_frame_with_eyes()

        if frame is None or len(eyes) == 0:
            return None

        (x, y, w, h) = eyes[0]
        eye_region = frame[y:y+h, x:x+w]

        height, width = eye_region.shape[:2]
        center_y, center_x = height // 2, width // 2
        roi = eye_region[center_y-height//4:center_y+height//4, 
                        center_x-width//4:center_x+width//4]
        
        avg_color = cv2.mean(roi)[:3]
        # Convert BGR to RGB
        return (avg_color[2], avg_color[1], avg_color[0])

    def release(self):
        self.cap.release()