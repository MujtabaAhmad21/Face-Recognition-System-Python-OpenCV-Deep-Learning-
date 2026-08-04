"""
encode_faces.py — CLI script to build encodings.pickle from known_faces/.

Walks known_faces/<name>/*.jpg, computes 128-d embeddings for each image,
and saves the result as a pickle file used by the recognition pipeline.

Usage:
    python encode_faces.py
"""

import pickle
import sys
from pathlib import Path

import cv2
import face_recognition

from recognition import detect_faces, encode_faces


KNOWN_FACES_DIR = Path("known_faces")
ENCODINGS_FILE = Path("encodings.pickle")


def build_encodings():
    """Scan known_faces/ and build the encodings database.

    Returns:
        tuple: (success: bool, message: str)
    """

    if not KNOWN_FACES_DIR.exists():
        msg = f"'{KNOWN_FACES_DIR}' directory not found — no faces to encode."
        print(msg)
        # Remove stale pickle so old faces don't persist
        if ENCODINGS_FILE.exists():
            ENCODINGS_FILE.unlink()
        return False, msg

    all_encodings = []
    all_names = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Walk each person's folder
    person_dirs = sorted([d for d in KNOWN_FACES_DIR.iterdir() if d.is_dir()])

    if not person_dirs:
        msg = f"No person folders found in '{KNOWN_FACES_DIR}/'. Cleared all encodings."
        print(msg)
        # Remove stale pickle so old faces don't persist
        if ENCODINGS_FILE.exists():
            ENCODINGS_FILE.unlink()
        return False, msg

    total_images = 0

    for person_dir in person_dirs:
        name = person_dir.name
        images = sorted([
            f for f in person_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ])

        if not images:
            print(f"  ⚠ {name}: no images found, skipping")
            continue

        print(f"  Processing {name}... ", end="", flush=True)
        count = 0

        for image_path in images:
            # Load image in RGB (face_recognition expects RGB)
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"\n    ⚠ Could not read {image_path.name}, skipping")
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect and encode
            locations = detect_faces(rgb, model="hog")
            encodings = encode_faces(rgb, locations)

            if not encodings:
                print(f"\n    ⚠ No face detected in {image_path.name}, skipping")
                continue

            # Use the first (largest/most prominent) face if multiple detected
            all_encodings.append(encodings[0])
            all_names.append(name)
            count += 1

        print(f"{count} image(s) encoded")
        total_images += count

    if total_images == 0:
        msg = "Error: No faces were successfully encoded."
        print(f"\n{msg}")
        return False, msg

    # Save to pickle
    data = {"encodings": all_encodings, "names": all_names}
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    unique_names = sorted(set(all_names))
    msg = f"Saved {total_images} encoding(s) for {len(unique_names)} person(s) to '{ENCODINGS_FILE}'"
    print(f"\n✓ {msg}")
    print(f"  People: {', '.join(unique_names)}")
    return True, msg


if __name__ == "__main__":
    print("Building face encodings...\n")
    success, message = build_encodings()
    if not success:
        sys.exit(1)

