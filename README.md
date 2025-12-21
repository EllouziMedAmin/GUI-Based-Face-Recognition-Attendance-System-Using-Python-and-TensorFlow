# GUI-Based Face Recognition Attendance System

## 1. Introduction

This project is a desktop application designed to automate the attendance tracking process using facial recognition technology. Built with **Python**, **TensorFlow**, and **OpenCV**, it features a user-friendly Graphical User Interface (GUI) that allows administrators to register students, build a biometric database, and mark attendance in real-time using a webcam.

## 2. Project Purpose

The primary goal of this project is to modernize traditional attendance systems. Manual methods (calling names or signing sheets) are time-consuming and prone to errors or manipulation (proxy attendance). This system aims to:

* **Eliminate Proxy Attendance:** Ensure the person marked present is physically there.
* **Save Time:** Automate the logging process so classes or meetings can start immediately.
* **Digitize Records:** Automatically generate CSV reports for easy data management.

## 3. How It Works

The system utilizes **InsightFace (ArcFace)** for high-accuracy face detection and recognition.

1. **Face Detection & Alignment:** When a face is seen by the camera, the system detects it and aligns it using facial landmarks.
2. **Embedding Generation:** The face is converted into a mathematical vector (embedding). This vector represents the unique features of the face.
3. **Registration:** During registration, multiple images are captured to create an average "reference" embedding for the student.
4. **Recognition:** In the live attendance mode, the system compares the current face's embedding against the database using **Cosine Similarity**. If the similarity score exceeds the threshold (0.60), the student is identified.
5. **Logging:** Valid identifications are logged into an `attendance.csv` file with the timestamp, preventing duplicate entries for the same day.

## 4. Impact

* **Efficiency:** Reduces the time spent on administrative tasks.
* **Accuracy:** Provides a reliable, biometric audit trail of attendance.
* **Accessibility:** The simple GUI makes advanced AI technology accessible to non-technical users (teachers or administrators).

---

## 5. Installation & Setup

### Prerequisites

* Python 3.11 (REQUIRED): This project relies on a specific pre-compiled wheel file for the insightface library which is built only for Python 3.11. Using other versions (like 3.10 or 3.12) will cause the installation to crash.
* A webcam

### Step 1: Clone the Repository

Download or clone this project to your local machine.
```bash
git clone https://github.com/EllouziMedAmin/GUI-Based-Face-Recognition-Attendance-System-Using-Python-and-TensorFlow.git
```

### Step 2: Install Dependencies

It is recommended to use a virtual environment.

```bash
# Create a virtual environment
python -3.11 -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

```

Install the required packages. Note that `insightface` can sometimes require C++ build tools. A specific wheel file is included in the project for Windows/Python 3.11 users.

```bash
pip install -r requirements.txt

```

*If you encounter issues installing `insightface`, you may need to install the provided wheel file manually:*

```bash
pip install "insightface-0.7.3-cp311-cp311-win_amd64.whl"

```

---

## 6. How to Run and Test the App

To start the application, run the main script:

```bash
python main.py

```

### Usage Instructions

The GUI has three main buttons corresponding to the workflow:

#### **Step 1: Register New Student**

1. Click **"1. Register New Student"**.
2. Enter the student's name in the popup dialog.
3. The webcam will open. Look at the camera.
4. Press **SPACE** to capture an image. You need to capture **5 images**.
5. The system will save these images in the `dataset/` folder.

#### **Step 2: Build Embeddings DB**

* *Note: You must do this after registering new students.*

1. Click **"2. Build Embeddings DB"**.
2. The system scans the `dataset/` folder, processes the face images, and saves the mathematical representations to `embeddings/students.npy`.
3. Wait for the "Success" message.

#### **Step 3: Start Attendance**

1. Click **"3. Start Attendance"**.
2. The webcam will open for live recognition.
3. When a registered face is detected with high confidence (Green Box), their name and a similarity score will appear.
4. **Check the Output:** Open `attendance.csv` in the project folder to see the logged entry (Name, Date, Time).
5. Press **'Q'** to quit the camera view.

---

## 7. Project Structure

* **`main.py`**: The entry point of the application. Handles the GUI and connects the logic components.
* **`face_recognition_tf.py`**: Contains the AI logic. Loads the InsightFace model and handles face analysis.
* **`attendance.py`**: Handles CSV operations (reading/writing attendance logs).
* **`Train_Custom_model.ipynb`**: (Optional) A notebook for training custom models if you wish to experiment with transfer learning using MobileNetV2 instead of the default InsightFace model.
* **`dataset/`**: Stores the raw images captured during registration.
* **`embeddings/`**: Stores the computed face embeddings (`students.npy`).