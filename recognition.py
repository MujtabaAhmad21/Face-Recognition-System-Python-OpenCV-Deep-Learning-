"""
recognition.py — Core face recognition pipeline.

Pure functions for detect → encode → match, fully decoupled from Flask.
Uses face_recognition (built on dlib's ResNet) to produce 128-d face embeddings
and compare them via Euclidean distance.
"""

import pickle
from pathlib import Path

import face_recognition
import numpy as np


# Named constant for the match threshold — easy to tune.
# Lower = stricter (fewer false positives, more "Unknown").
# Higher = more lenient (more matches, risk of mis-identification).
MATCH_THRESHOLD = 0.6


def detect_faces(frame, model="hog"):
    """Detect face locations in an RGB frame.

    Args:
        frame: RGB image (numpy array, H×W×3).
        model: "hog" (fast, CPU) or "cnn" (accurate, needs GPU).
               Use "hog" on Apple Silicon — "cnn" is too slow without CUDA.

    Returns:
        List of face location tuples: (top, right, bottom, left).
    """
    return face_recognition.face_locations(frame, model=model)


def encode_faces(frame, locations):
    """Compute 128-d face embeddings for detected face locations.

    Args:
        frame: RGB image (numpy array, H×W×3).
        locations: List of (top, right, bottom, left) tuples from detect_faces().

    Returns:
        List of 128-d numpy arrays — one encoding per face.
    """
    return face_recognition.face_encodings(frame, locations)


def match_faces(encodings, known_encodings, known_names, threshold=MATCH_THRESHOLD):
    """Match detected face encodings against the known database.

    Uses face_distance (Euclidean distance in 128-d space) to find the
    closest known face. Returns the name and confidence percentage.

    Args:
        encodings: List of 128-d encodings from the current frame.
        known_encodings: List of 128-d encodings from the database.
        known_names: Corresponding names for known_encodings.
        threshold: Maximum distance to consider a match (default 0.6).

    Returns:
        List of dicts: [{"name": str, "confidence": int, "distance": float}, ...]
    """
    if not encodings:
        return []

    if not known_encodings:
        return [{"name": "Unknown", "confidence": 0, "distance": 1.0} for _ in encodings]

    results = []
    for encoding in encodings:
        distances = face_recognition.face_distance(known_encodings, encoding)
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])

        # Convert distance to confidence percentage (0-100%)
        # Distance 0.0 -> 100% confidence, Distance 0.6 -> ~50-60% confidence
        confidence = max(0, min(99, int(round((1.0 - (best_distance / 1.2)) * 100))))

        if best_distance <= threshold:
            results.append({
                "name": known_names[best_idx],
                "confidence": max(50, confidence),
                "distance": round(best_distance, 3)
            })
        else:
            results.append({
                "name": "Unknown",
                "confidence": confidence,
                "distance": round(best_distance, 3)
            })

    return results



def load_encodings(path="encodings.pickle"):
    """Load pre-computed encodings from a pickle file.

    Args:
        path: Path to the pickle file.

    Returns:
        Tuple of (encodings_list, names_list).
        Returns ([], []) if the file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        return [], []

    with open(path, "rb") as f:
        data = pickle.load(f)

    return data.get("encodings", []), data.get("names", [])
