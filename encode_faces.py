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
    """Scan known_faces/ and build the encodings database."""

    if not KNOWN_FACES_DIR.exists():
        print(f"Error: '{KNOWN_FACES_DIR}' directory not found.")
        print("Create it and add sub-folders with person names containing their photos.")
        sys.exit(1)

    all_encodings = []
    all_names = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Walk each person's folder
    person_dirs = sorted([d for d in KNOWN_FACES_DIR.iterdir() if d.is_dir()])

    if not person_dirs:
        print(f"Warning: No sub-folders found in '{KNOWN_FACES_DIR}/'.")
        print("Add folders named after each person, containing their photos.")
        sys.exit(1)

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
        print("\nError: No faces were successfully encoded.")
        sys.exit(1)

    # Save to pickle
    data = {"encodings": all_encodings, "names": all_names}
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    unique_names = sorted(set(all_names))
    print(f"\n✓ Saved {total_images} encoding(s) for {len(unique_names)} person(s) to '{ENCODINGS_FILE}'")
    print(f"  People: {', '.join(unique_names)}")


if __name__ == "__main__":
    print("Building face encodings...\n")
    build_encodings()
