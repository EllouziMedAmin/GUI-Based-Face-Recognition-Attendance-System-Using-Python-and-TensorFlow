import cv2
import os
import shutil
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from sklearn.metrics.pairwise import cosine_similarity
from face_recognition_tf import get_face_analysis
from attendance import mark_attendance

# --- Configuration ---
DATASET_DIR = "dataset/students"
EMBEDDINGS_FILE = "embeddings/students.npy"
THRESHOLD = 0.60

# Ensure directories exist
os.makedirs("embeddings", exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# --- Logic Functions ---

def normalize_vector(embedding):
    """Normalizes the vector as per the InsightFace example."""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def register_student_logic(name):
    """
    Captures images for a specific student name.
    """
    student_dir = os.path.join(DATASET_DIR, name)
    if os.path.exists(student_dir):
        shutil.rmtree(student_dir) # Clear old bad data
    os.makedirs(student_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    
    # Check if camera opened
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam.")
        return

    print(f"--- REGISTRATION FOR {name} ---")
    
    while count < 5:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        
        # Check if face is visible
        face_obj = get_face_analysis(frame)
        
        if face_obj is not None:
            # Draw box for feedback
            x1, y1, x2, y2 = map(int, face_obj.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Show text on frame
        cv2.putText(frame, f"Registering: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Captured: {count}/5 (Press SPACE)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Register Student", frame)

        if cv2.waitKey(1) & 0xFF == 32: # Space bar
            if face_obj is not None:
                img_path = f"{student_dir}/img{count}.jpg"
                cv2.imwrite(img_path, frame)
                print(f"Captured {count+1}/5")
                count += 1
            else:
                print("No face detected!")

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Success", f"Registration completed for {name}.")

def build_embeddings_logic():
    """
    Processes the dataset and rebuilds the embeddings file.
    """
    database = {}

    if not os.path.exists(DATASET_DIR):
        messagebox.showwarning("Warning", "No dataset found.")
        return

    student_names = os.listdir(DATASET_DIR)
    if not student_names:
        messagebox.showwarning("Warning", "No students found in dataset.")
        return

    for name in student_names:
        student_path = os.path.join(DATASET_DIR, name)
        if not os.path.isdir(student_path): continue
            
        embeddings = []
        for img_name in os.listdir(student_path):
            img_path = os.path.join(student_path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue

            face_obj = get_face_analysis(img)

            if face_obj is not None:
                norm_emb = normalize_vector(face_obj.embedding)
                embeddings.append(norm_emb)

        if embeddings:
            avg_emb = np.mean(embeddings, axis=0)
            database[name] = normalize_vector(avg_emb)
            print(f"Registered: {name} ({len(embeddings)} samples)")

    np.save(EMBEDDINGS_FILE, database)
    messagebox.showinfo("Success", "Database built successfully!")

def recognize_attendance_logic():
    """
    Runs the live recognition loop.
    """
    if not os.path.exists(EMBEDDINGS_FILE):
        messagebox.showerror("Error", "No database found. Please build embeddings first.")
        return

    database = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam.")
        return

    print("--- RECOGNITION STARTED ---")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        
        face_obj = get_face_analysis(frame)

        if face_obj is not None:
            curr_emb = normalize_vector(face_obj.embedding)
            
            best_score = 0
            best_match = "Unknown"
            
            for name, db_emb in database.items():
                score = cosine_similarity([curr_emb], [db_emb])[0][0]
                if score > best_score:
                    best_score = score
                    best_match = name
            
            x1, y1, x2, y2 = map(int, face_obj.bbox)
            
            if best_score > THRESHOLD:
                color = (0, 255, 0) # Green
                label = f"{best_match} ({best_score:.2f})"
                mark_attendance(best_match)
            else:
                color = (0, 0, 255) # Red
                label = f"Unknown ({best_score:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.putText(frame, "Press 'Q' to Exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Attendance System", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Tkinter GUI ---

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("400x350")
        
        # Title Label
        title_lbl = tk.Label(root, text="Attendance System", font=("Helvetica", 16, "bold"))
        title_lbl.pack(pady=20)

        # Buttons
        btn_register = tk.Button(root, text="1. Register New Student", font=("Arial", 12), width=25, command=self.on_register)
        btn_register.pack(pady=10)

        btn_build = tk.Button(root, text="2. Build Embeddings DB", font=("Arial", 12), width=25, command=self.on_build)
        btn_build.pack(pady=10)

        btn_mark = tk.Button(root, text="3. Start Attendance", font=("Arial", 12), width=25, bg="#d9ffcc", command=self.on_recognize)
        btn_mark.pack(pady=10)

        btn_exit = tk.Button(root, text="Exit", font=("Arial", 12), width=25, fg="red", command=root.quit)
        btn_exit.pack(pady=20)

    def on_register(self):
        name = simpledialog.askstring("Input", "Enter student name:")
        if name:
            name = name.strip().lower()
            if name:
                register_student_logic(name)
            else:
                messagebox.showwarning("Input Error", "Name cannot be empty.")

    def on_build(self):
        # Disable interaction slightly or show info
        build_embeddings_logic()

    def on_recognize(self):
        recognize_attendance_logic()

if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    app = AttendanceApp(root)
    # Start the GUI event loop
    root.mainloop()