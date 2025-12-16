import cv2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from face_recognition_tf import extract_face, get_embedding
from attendance import mark_attendance

DATASET_DIR = "dataset/students"
EMBEDDINGS_FILE = "embeddings/students.npy"
THRESHOLD = 0.6

os.makedirs("embeddings", exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

def register_student():
    name = input("Enter student name: ").strip().lower()
    student_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(student_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    print("Press SPACE to capture images (5 images recommended)")
    while count < 5:
        ret, frame_row = cap.read()
        frame =cv2.flip(frame_row, 1)
        cv2.imshow("Register Face", frame)

        if cv2.waitKey(1) & 0xFF == 32:
            face = extract_face(frame)
            if face is not None:
                img_path = f"{student_dir}/img{count}.jpg"
                cv2.imwrite(img_path, face)
                print("Saved:", img_path)
                count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Registration completed.")

def build_embeddings():
    database = {}

    for name in os.listdir(DATASET_DIR):
        embeddings = []
        for img_name in os.listdir(f"{DATASET_DIR}/{name}"):
            img = cv2.imread(f"{DATASET_DIR}/{name}/{img_name}")
            face = extract_face(img)
            if face is not None:
                emb = get_embedding(face)
                embeddings.append(emb)

        if embeddings:
            database[name] = np.mean(embeddings, axis=0)

    np.save(EMBEDDINGS_FILE, database)
    print("Embeddings updated.")

def recognize_and_mark():
    if not os.path.exists(EMBEDDINGS_FILE):
        print("No embeddings found. Register students first.")
        return

    database = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()

    cap = cv2.VideoCapture(0)
    print("Press Q to quit")

    while True:
        ret, frame_raw = cap.read()
        frame = cv2.flip(frame_raw, 1)
        face = extract_face(frame)

        if face is not None:
            emb = get_embedding(face)
            for name, db_emb in database.items():
                score = cosine_similarity([emb], [db_emb])[0][0]
                if score > THRESHOLD:
                    marked = mark_attendance(name)
                    label = f"{name} {'Marked' if marked else 'Already Marked'}"
                    cv2.putText(frame, label, (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

while True:
    print("\n--- Face Attendance System ---")
    print("1. Register Student")
    print("2. Build Embeddings")
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
