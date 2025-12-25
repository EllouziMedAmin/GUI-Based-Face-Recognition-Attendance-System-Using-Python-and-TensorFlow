import cv2
import os
import shutil
import numpy as np
import tkinter as tk
import pandas as pd  # Added for reading the CSV
from datetime import datetime # Added for time handling
from tkinter import simpledialog, messagebox
from sklearn.metrics.pairwise import cosine_similarity
from face_recognition_tf import get_face_analysis
from attendance import mark_attendance

# --- Configuration ---
DATASET_DIR = "dataset/students"
EMBEDDINGS_FILE = "embeddings/students.npy"
THRESHOLD = 0.60
ATTENDANCE_FILE = "attendance.csv" # Ensure this matches your attendance.py file name

# Ensure directories exist
os.makedirs("embeddings", exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# --- Helper Functions for Display & Logic ---

def normalize_vector(embedding):
    """Normalizes the vector as per the InsightFace example."""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def load_todays_attendance():
    """
    Reads the CSV file and loads today's existing records into memory.
    Returns a dictionary: { 'Name': '08:30:00' }
    """
    cache = {}
    if not os.path.exists(ATTENDANCE_FILE):
        return cache

    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        # Filter for rows where Date matches today
        if not df.empty and "Date" in df.columns:
            todays_records = df[df["Date"] == today_str]
            for index, row in todays_records.iterrows():
                cache[row["Name"]] = str(row["Time"])
                
        print(f"Loaded {len(cache)} existing records for today.")
    except Exception as e:
        print(f"Warning: Could not load existing attendance: {e}")
        
    return cache

def draw_hud(img, bbox, name, score, time_marked, is_new):
    """
    Draws a professional info box (HUD) around the face.
    """
    x1, y1, x2, y2 = bbox
    
    # Color: Bright Green if new, Darker Green if already marked
    color_bg = (0, 200, 0) if is_new else (0, 120, 0)
    color_text = (255, 255, 255)
    
    # Draw Bounding Box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Prepare Text Labels
    label_name = f"{name.upper()} ({int(score*100)}%)"
    label_time = f"Marked: {time_marked}"
    
    # Calculate Text Size
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 1
    (w1, h1), _ = cv2.getTextSize(label_name, font, scale, thick)
    (w2, h2), _ = cv2.getTextSize(label_time, font, scale, thick)
    
    box_w = max(w1, w2) + 20
    box_h = h1 + h2 + 25
    
    # Draw Background Rectangle (Above Face)
    # Ensure it stays on screen (y > 0)
    start_y = max(y1 - box_h, 0)
    cv2.rectangle(img, (x1, start_y), (x1 + box_w, y1), color_bg, -1)
    
    # Put Text
    cv2.putText(img, label_name, (x1 + 10, y1 - h2 - 15), font, scale, color_text, thick+1)
    cv2.putText(img, label_time, (x1 + 10, y1 - 10), font, scale, color_text, thick)

# --- Core Logic Functions ---

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
    
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam.")
        return

    print(f"--- REGISTRATION FOR {name} ---")
    
    while count < 5:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        face_obj = get_face_analysis(frame)
        
        if face_obj is not None:
            x1, y1, x2, y2 = map(int, face_obj.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
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
    Runs the live recognition loop with Enhanced Display & Caching.
    """
    if not os.path.exists(EMBEDDINGS_FILE):
        messagebox.showerror("Error", "No database found. Please build embeddings first.")
        return

    database = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
    
    # --- LOAD MEMORY ---
    # Load who has already been marked today from the CSV
    attendance_cache = load_todays_attendance()

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
            
            # Find best match
            for name, db_emb in database.items():
                score = cosine_similarity([curr_emb], [db_emb])[0][0]
                if score > best_score:
                    best_score = score
                    best_match = name
            
            # Get coordinates
            bbox = map(int, face_obj.bbox)
            x1, y1, x2, y2 = bbox
            
            if best_score > THRESHOLD:
                display_time = ""
                is_new_mark = False

                # --- CHECK MEMORY ---
                if best_match in attendance_cache:
                    # ALREADY MARKED: Retrieve the old time from cache
                    display_time = attendance_cache[best_match]
                else:
                    # NEW MARK: Save to CSV and update cache
                    mark_attendance(best_match) 
                    
                    # Update local cache immediately
                    current_time = datetime.now().strftime("%H:%M:%S")
                    attendance_cache[best_match] = current_time
                    display_time = current_time
                    is_new_mark = True

                # DRAW ENHANCED BOX
                draw_hud(frame, (x1, y1, x2, y2), best_match, best_score, display_time, is_new_mark)

            else:
                # UNKNOWN
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
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
        build_embeddings_logic()

    def on_recognize(self):
        recognize_attendance_logic()

if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    app = AttendanceApp(root)
    # Start the GUI event loop
    root.mainloop()