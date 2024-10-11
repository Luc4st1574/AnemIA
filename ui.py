import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import threading
import cv2

class ControlPanel(tk.Tk):
    def __init__(self, eye_detector, model):
        super().__init__()
        self.title("AnemIA")
        self.eye_detector = eye_detector
        self.model = model
        self.running = True

        # Increase window size
        self.geometry("1000x700")  # Width x Height

        # Apply dark mode theme colors
        self.configure(bg='#2C2F33')  # Dark background

        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initialize video display
        self.video_label = tk.Label(self.left_frame, bg='#2C2F33')
        self.video_label.pack(expand=True)  # This centers the video by expanding in both directions

        # Initialize the thread for updating video
        self.update_thread = threading.Thread(target=self.update_video_thread)
        self.update_thread.start()

    def create_widgets(self):
        """Create and layout the GUI components using Frames."""
        # Create a main frame to hold left and right frames
        main_frame = tk.Frame(self, bg='#2C2F33')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left Frame for Video
        self.left_frame = tk.Frame(main_frame, width=600, height=600, bg='#2C2F33')
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right Frame for Controls
        self.right_frame = tk.Frame(main_frame, width=400, height=600, padx=20, pady=20, bg='#23272A')
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Configure grid in right_frame for better layout management
        self.right_frame.columnconfigure(0, weight=1)

        # === Add Logo to Right Frame ===
        try:
            image_path = "AnemIA\\resources\\logo.png"
            load = Image.open(image_path)
            load = load.resize((200, 200), Image.LANCZOS)
            self.logo_image = ImageTk.PhotoImage(load)

            logo_label = tk.Label(self.right_frame, image=self.logo_image, bg='#23272A')
            logo_label.grid(row=0, column=0, pady=(0, 20), sticky='n')  # Place logo in row 0, centered
        except Exception as e:
            print(f"Error loading image: {e}")
            logo_label = tk.Label(self.right_frame, text="Logo Here", font=("Arial", 16), bg='#23272A', fg='white')
            logo_label.grid(row=0, column=0, pady=(0, 20), sticky='n')  # Error message in case logo fails

        # Add a rectangular image below the logo
        try:
            rect_image_path = "AnemIA\\resources\\anemIA.png"
            rect_load = Image.open(rect_image_path)
            rect_load = rect_load.resize((300, 75), Image.LANCZOS)  # Adjust the size as needed
            self.rect_image = ImageTk.PhotoImage(rect_load)

            rect_label = tk.Label(self.right_frame, image=self.rect_image, bg='#23272A')
            rect_label.grid(row=1, column=0, pady=(0, 20), sticky='n')  # Place rectangle image in row 1, centered
        except Exception as e:
            print(f"Error loading rectangle image: {e}")
            rect_label = tk.Label(self.right_frame, text="Rectangle Image Here", font=("Arial", 16), bg='#23272A', fg='white')
            rect_label.grid(row=1, column=0, pady=(0, 20), sticky='n')  # Error message in case rectangle fails

        # === Start next widgets after the image ===
        # Sex Input using Radio Buttons
        sex_label = tk.Label(self.right_frame, text="Sexo:", font=("Helvetica", 12), bg='#23272A', fg='white')
        sex_label.grid(row=2, column=0, sticky='w', pady=(10, 5))  # Shift to row 2

        self.sex_var = tk.IntVar(value=0)  # Default to Male (0)
        male_radio = tk.Radiobutton(self.right_frame, text="Hombre", variable=self.sex_var, value=0, font=("Helvetica", 12), bg='#23272A', fg='white', selectcolor='#99AAB5')
        female_radio = tk.Radiobutton(self.right_frame, text="Mujer", variable=self.sex_var, value=1, font=("Helvetica", 12), bg='#23272A', fg='white', selectcolor='#99AAB5')

        male_radio.grid(row=3, column=0, sticky='w')  # Shift to row 3
        female_radio.grid(row=4, column=0, sticky='w')  # Shift to row 4

        # Hemoglobin Level Input using Scale
        hb_label = tk.Label(self.right_frame, text="Nivel de Hemoglobina (Hb):", font=("Helvetica", 12), bg='#23272A', fg='white')
        hb_label.grid(row=5, column=0, sticky='w', pady=(10, 5))  # Shift to row 5

        self.hb_var = tk.DoubleVar(value=10)  # Default value
        self.hb_scale = tk.Scale(self.right_frame, from_=1, to=20, orient=tk.HORIZONTAL, variable=self.hb_var, font=("Helvetica", 12), bg='#23272A', fg='white', troughcolor='#7289DA', highlightbackground='#99AAB5')
        self.hb_scale.grid(row=6, column=0, sticky='ew', pady=(0, 5))  # Shift to row 6

        # Label to display the selected Hb value
        self.hb_value_label = tk.Label(self.right_frame, text=f"Hemoglobina: {self.hb_var.get()}", font=("Helvetica", 12), bg='#23272A', fg='white')
        self.hb_value_label.grid(row=7, column=0, sticky='w')  # Shift to row 7

        # Update the Hb value label dynamically
        self.hb_scale.bind("<Motion>", self.update_hb_value)

        # RGB Label
        self.rgb_label = tk.Label(self.right_frame, text="RGB Values: (0, 0, 0)", font=("Helvetica", 12), bg='#23272A', fg='white')
        self.rgb_label.grid(row=8, column=0, sticky='w', pady=(10, 5))  # Shift to row 8

        # Predict Button with hover effect
        predict_button = tk.Button(self.right_frame, text="Predecir Anemia", command=self.predict_anemia, font=("Helvetica", 12), bg='#7289DA', fg='white', activebackground='#99AAB5', activeforeground='black', relief=tk.FLAT)
        predict_button.grid(row=9, column=0, pady=(10, 20), sticky='ew')  # Shift to row 9

        # Add hover effect for button
        predict_button.bind("<Enter>", lambda e: predict_button.config(bg='#99AAB5', fg='black'))
        predict_button.bind("<Leave>", lambda e: predict_button.config(bg='#7289DA', fg='white'))

    def update_video_thread(self):
        """Thread function to continuously update video and RGB values."""
        while self.running:
            # Get frame and eye coordinates from the eye detector
            frame, eye_coords = self.eye_detector.get_frame_with_eyes()

            if frame is not None and self.running:
                # Check if eyes were detected (i.e., eye_coords is not empty)
                if len(eye_coords) > 0:
                    for (x, y, w, h) in eye_coords:
                        # Draw a rectangle around the detected eye region
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle, 2 px thick

                # Detect eyes and get RGB values
                avg_rgb = self.eye_detector.capture_eye_rgb_auto()
                if avg_rgb is not None:
                    self.current_rgb = tuple(int(c) for c in avg_rgb[:3])  # Convert to integers
                    # Update RGB label in the main thread
                    self.rgb_label.after(0, lambda: self.rgb_label.config(
                        text=f"RGB Values: {self.current_rgb}"
                    ))

                # Convert frame from BGR to RGB color space
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert frame to ImageTk format
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)

                # Update the video label in the main thread
                self.video_label.after(0, lambda: self.video_label.config(image=imgtk))
                self.video_label.imgtk = imgtk  # Keep a reference to avoid garbage collection

            if self.running:
                self.after(30)  # Small delay to prevent high CPU usage

    def stop_video_capture(self):
        """Stop the video capture and release resources."""
        if self.running:
            self.running = False
            self.update_thread.join(timeout=1)  # Ensure the video thread stops
            self.eye_detector.release()  # Release the video capture from the eye detector
            print("Video capture stopped.")

    def update_hb_value(self, event=None):
        """Update the Hb value label as the scale changes."""
        self.hb_value_label.config(text=f"Selected Hb: {self.hb_var.get():.1f}")

    def predict_anemia(self):
        """Handle the prediction logic and display a simple message box with the result."""
        try:
            sex = self.sex_var.get()
            hb = self.hb_var.get()

            avg_rgb = self.eye_detector.capture_eye_rgb_auto()
            if avg_rgb is None:
                raise ValueError("Failed to capture RGB values.")

            R, G, B = avg_rgb

            input_data = np.array([[sex, R, G, B, hb]])
            details = self.model.predict(input_data)

            if details['prediction'] == 'Anemia':
                message = "Se ha detectado anemia. Por favor, consulte a un médico para obtener más información y tratamiento adecuado."
                messagebox.showwarning("Resultado de la Predicción", message)
            else:
                message = "¡Felicidades! No se ha detectado anemia. Mantenga una dieta saludable y hábitos de vida equilibrados."
                messagebox.showinfo("Resultado de la Predicción", message)

        except ValueError as ve:
            messagebox.showerror("Error de Entrada", f"Entrada inválida: {ve}")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error durante la predicción.\n{e}")

    def on_closing(self):
        """Handle the window closing event."""
        self.stop_video_capture()  # Use stop_video_capture for cleanup
        self.destroy()
