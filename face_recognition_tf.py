import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# Initialize InsightFace (ArcFace model: buffalo_l)
# This downloads the model once and then runs offline.
print("Loading InsightFace model...")
try:
    # Try GPU (ctx_id=0)
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Model loaded on GPU.")
except Exception:
    # Fallback to CPU (ctx_id=-1)
    print("GPU not found. Switching to CPU...")
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))

def get_face_analysis(img_bgr):
    """
    Scans the image for faces using InsightFace.
    Returns the single 'best' face object (largest area) containing:
    - .embedding (vector)
    - .bbox (coordinates)
    """
    if img_bgr is None: 
        return None
    
    # InsightFace works directly on BGR images (no conversion needed)
    faces = app.get(img_bgr)
    
    if not faces:
        return None
        
    # Sort faces by area (width * height) to pick the main person
    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    
    return best_face