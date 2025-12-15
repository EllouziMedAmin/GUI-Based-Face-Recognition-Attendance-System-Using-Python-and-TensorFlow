import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

detector = MTCNN()
embedder = FaceNet()

def extract_face(img, padding=0.3):
    faces = detector.detect_faces(img)
    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]['box']

    pad = int(w * padding)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad)
    y2 = min(img.shape[0], y + h + pad)

    face = img[y1:y2, x1:x2]
    face = cv2.resize(face, (160, 160))
    return face

def get_embedding(face):
    face = face.astype("float32")
    face = (face - 127.5) / 128.0
    embedding = embedder.embeddings([face])[0]
    embedding = embedding / np.linalg.norm(embedding)
    return embedding
