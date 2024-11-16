import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import threading
import cv2
import os
import csv
import sys

class ControlPanel(tk.Tk):
    def __init__(self, eye_detector, model):
        super().__init__()
        self.title("AnemIA")
        self.iconbitmap("AnemIA\\logo.ico") 
        self.eye_detector = eye_detector
        self.model = model
        self.running = True
        self.current_rgb = (0, 0, 0)

        self.geometry("1000x700")
        self.configure(bg='#2C2F33')

        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initialize video display
        self.video_label = tk.Label(self.left_frame, bg='#2C2F33')
        self.video_label.pack(expand=True)

        # Initialize the thread for updating video
        self.update_thread = threading.Thread(target=self.update_video_thread)
        self.update_thread.start()

    def create_widgets(self):
        """Create and layout the GUI components using Frames."""
        # Main frame to hold left and right frames
        main_frame = tk.Frame(self, bg='#2C2F33')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left Frame for Video
        self.left_frame = tk.Frame(main_frame, width=600, height=600, bg='#2C2F33')
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right Frame for Controls
        self.right_frame = tk.Frame(main_frame, width=400, height=600, padx=20, pady=20, bg='#23272A')
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Configure grid in right_frame
        self.right_frame.columnconfigure(0, weight=1)

        # Patient Name Input
        tk.Label(self.right_frame, text="Nombre del paciente:", font=("Helvetica", 12), bg='#23272A', fg='white').grid(row=0, column=0, sticky='w')
        self.patient_name_entry = tk.Entry(self.right_frame, font=("Helvetica", 12))
        self.patient_name_entry.grid(row=1, column=0, pady=(5, 20), sticky='ew')

        # Label for predicted Hb value
        self.hb_value_label = tk.Label(self.right_frame, text="Hemoglobina estimada: --", font=("Helvetica", 12), bg='#23272A', fg='white')
        self.hb_value_label.grid(row=2, column=0, pady=(10, 20), sticky='w')

        # RGB Label
        self.rgb_label = tk.Label(self.right_frame, text="RGB Values: (0, 0, 0)", font=("Helvetica", 12), bg='#23272A', fg='white')
        self.rgb_label.grid(row=3, column=0, pady=(10, 20), sticky='w')

        # Sex Input
        tk.Label(self.right_frame, text="Sexo:", font=("Helvetica", 12), bg='#23272A', fg='white').grid(row=4, column=0, sticky='w')
        self.sex_var = tk.IntVar(value=0)  # Default: Male (0)
        tk.Radiobutton(self.right_frame, text="Hombre", variable=self.sex_var, value=0, font=("Helvetica", 12), bg='#23272A', fg='white', selectcolor='#99AAB5').grid(row=5, column=0, sticky='w')
        tk.Radiobutton(self.right_frame, text="Mujer", variable=self.sex_var, value=1, font=("Helvetica", 12), bg='#23272A', fg='white', selectcolor='#99AAB5').grid(row=6, column=0, sticky='w')

        # Predict Button
        predict_button = tk.Button(self.right_frame, text="Predecir Anemia", command=self.predict_anemia, font=("Helvetica", 12), bg='#7289DA', fg='white')
        predict_button.grid(row=7, column=0, pady=20, sticky='ew')

        # Reset Button
        reset_button = tk.Button(self.right_frame, text="Nueva Predicción", command=self.reset_ui, font=("Helvetica", 12), bg='#99AAB5', fg='white')
        reset_button.grid(row=8, column=0, pady=10, sticky='ew')

        # Close Button
        close_button = tk.Button(self.right_frame, text="Cerrar", command=self.on_closing, font=("Helvetica", 12), bg='#FF5C5C', fg='white')
        close_button.grid(row=9, column=0, pady=10, sticky='ew')

    def update_video_thread(self):
        """Thread function to continuously update video and RGB values."""
        while self.running:
            frame, eye_coords = self.eye_detector.get_frame_with_eyes()

            if frame is not None and self.running:
                if len(eye_coords) > 0:
                    for (x, y, w, h) in eye_coords:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                avg_rgb = self.eye_detector.capture_eye_rgb_auto()
                if avg_rgb is not None:
                    self.current_rgb = tuple(int(c) for c in avg_rgb[:3])
                    self.rgb_label.after(0, lambda: self.rgb_label.config(
                        text=f"RGB Values: {self.current_rgb}"
                    ))

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.after(0, lambda: self.video_label.config(image=imgtk))
                self.video_label.imgtk = imgtk

    def predict_anemia(self):
        """Handle prediction and update Hb label."""
        try:
            name = self.patient_name_entry.get().strip()
            if not name:
                raise ValueError("El nombre del paciente no puede estar vacío.")
            sex = self.sex_var.get()
            R, G, B = self.current_rgb

            input_data = np.array([[sex, R, G, B]])
            details = self.model.predict(input_data)

            predicted_hb = details['predicted_Hb']
            predicted_anemia = details['predicted_Anemic']
            self.hb_value_label.config(text=f"Hemoglobina estimada: {predicted_hb:.2f} g/dL")

            message = ("Se ha detectado anemia. Consulte a un médico." if predicted_anemia == 'Anemia' 
                    else "¡Sin anemia detectada! Mantenga hábitos saludables.")
            messagebox.showinfo("Resultado de la Predicción", message)

            # Save results to CSV
            self.save_to_csv(name, sex, R, G, B, predicted_hb, predicted_anemia)
        except ValueError as ve:
            messagebox.showerror("Error de Entrada", f"Entrada inválida: {ve}")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error durante la predicción.\n{e}")

    def save_to_csv(self, name, sex, R, G, B, hb, anemia):
        """Save prediction data to CSV file."""
        file_path = os.path.join("AnemIA\\Exported", "exported_anemia_data.csv")
        os.makedirs("AnemIA\\Exported", exist_ok=True)
        file_exists = os.path.isfile(file_path)

        # Convert 'Anemia' to 'Yes' and 'No'
        anemia_status = "Yes" if anemia == 'Anemia' else "No"

        with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["Nombre", "Sexo", "R", "G", "B", "Hemoglobina", "Anemia"])
            writer.writerow([name, "Hombre" if sex == 0 else "Mujer", R, G, B, f"{hb:.2f}", anemia_status])

            

    def reset_ui(self):
        """Reset the UI for a new prediction."""
        self.patient_name_entry.delete(0, tk.END)
        self.hb_value_label.config(text="Hemoglobina estimada: --")
        self.rgb_label.config(text="RGB Values: (0, 0, 0)")

    def on_closing(self):
        """Handle cleanup and window close."""
        self.running = False  # Stop the video thread
        self.eye_detector.release()  # Release camera resources
        self.destroy()  # Close the application window
        print("Resources released and application closed.")
        sys.exit(0)  # Terminate the Python process