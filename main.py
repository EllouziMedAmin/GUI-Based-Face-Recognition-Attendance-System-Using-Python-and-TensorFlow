import cv2
import os
import shutil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from face_recognition_tf import get_face_analysis
from attendance import mark_attendance

DATASET_DIR = "dataset/students"
EMBEDDINGS_FILE = "embeddings/students.npy"

# Threshold Guide:
# 0.75: Very Strict
# 0.65: Balanced
# 0.55: Permissive
THRESHOLD = 0.60

os.makedirs("embeddings", exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

def normalize_vector(embedding):
    """Normalizes the vector as per the InsightFace example."""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def register_student():
    name = input("Enter student name: ").strip().lower()
    if not name: return
    
    student_dir = os.path.join(DATASET_DIR, name)
    if os.path.exists(student_dir):
        shutil.rmtree(student_dir) # Clear old bad data
    os.makedirs(student_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    print("--- REGISTRATION ---")
    print("Press SPACE to capture (5 images).")
    
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
        
        cv2.imshow("Register (Press SPACE)", frame)

        if cv2.waitKey(1) & 0xFF == 32: 
            if face_obj is not None:
                # Save the FULL FRAME (InsightFace handles cropping internally)
                img_path = f"{student_dir}/img{count}.jpg"
                cv2.imwrite(img_path, frame)
                print(f"Captured {count+1}/5")
                count += 1
            else:
                print("No face detected! Adjust lighting.")

    cap.release()
    cv2.destroyAllWindows()
    print("Registration completed.")

def build_embeddings():
    print("Building database...")
    database = {}

    if not os.path.exists(DATASET_DIR):
        print("No dataset found.")
        return

    for name in os.listdir(DATASET_DIR):
        student_path = os.path.join(DATASET_DIR, name)
        if not os.path.isdir(student_path): continue
            
        embeddings = []
        for img_name in os.listdir(student_path):
            img_path = os.path.join(student_path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue

            # Get embedding from full image
            face_obj = get_face_analysis(img)

            if face_obj is not None:
                # Normalize before saving
                norm_emb = normalize_vector(face_obj.embedding)
                embeddings.append(norm_emb)
            else:
                print(f"Warning: No face in {img_name}")

        if embeddings:
            # Average the normalized embeddings
            avg_emb = np.mean(embeddings, axis=0)
            database[name] = normalize_vector(avg_emb) # Normalize again after averaging
            print(f"Registered: {name} ({len(embeddings)} samples)")
        else:
            print(f"Skipped: {name} (No valid faces)")

    np.save(EMBEDDINGS_FILE, database)
    print("Database built successfully.")

def recognize_and_mark():
    if not os.path.exists(EMBEDDINGS_FILE):
        print("No database found. Run Option 2.")
        return

    database = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
    cap = cv2.VideoCapture(0)
    
    print("--- RECOGNITION ---")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        
        # 1. Detect Face
        face_obj = get_face_analysis(frame)

        if face_obj is not None:
            # 2. Get Embedding & Normalize
            curr_emb = normalize_vector(face_obj.embedding)
            
            best_score = 0
            best_match = "Unknown"
            
            # 3. Compare with Database
            for name, db_emb in database.items():
                score = cosine_similarity([curr_emb], [db_emb])[0][0]
                if score > best_score:
                    best_score = score
                    best_match = name
            
            # 4. Decide
            x1, y1, x2, y2 = map(int, face_obj.bbox)
            
            if best_score > THRESHOLD:
                color = (0, 255, 0) # Green
                label = f"{best_match} ({best_score:.2f})"
                mark_attendance(best_match)
            else:
                color = (0, 0, 255) # Red
                label = f"Unknown ({best_score:.2f})"

            # Draw visual feedback
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

while True:
    print("\n--- InsightFace Attendance System ---")
    print("1. Register Student")
    print("2. Build Embeddings (RESET DB)")
    print("3. Mark Attendance")
    print("4. Exit")

    choice = input("Choose option: ")

    if choice == "1":
        register_student()
    elif choice == "2":
        build_embeddings()
    elif choice == "3":
        recognize_and_mark()
    else:
        break