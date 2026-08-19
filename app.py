"""
app.py — Flask entry point with MJPEG streaming for real-time face recognition.

Routes:
    /            — Main page with live annotated video feed
    /video_feed  — MJPEG stream (multipart/x-mixed-replace)
    /enroll      — In-browser enrollment page (GET/POST)
    /capture_frame — Returns current frame as JPEG for enrollment
"""

import os
import socket
import sys
import time
import threading

import io

# On macOS, prevent OpenCV from requesting camera permission from a background thread.
# This avoids the "can not spin main run loop from other thread" crash.
# The user must have already granted terminal permission (e.g. by running test_camera.py).
if sys.platform == "darwin":
    os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

import shutil

import cv2
import numpy as np
from flask import Flask, Response, render_template, request, jsonify, send_from_directory
from imutils.video import FPS
from werkzeug.serving import make_server

from recognition import (
    detect_faces,
    encode_faces,
    match_faces,
    load_encodings,
    MATCH_THRESHOLD,
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Camera singleton — guarded to avoid double-init from Flask reloader
# ---------------------------------------------------------------------------
camera_lock = threading.Lock()
camera_read_lock = threading.Lock()
camera = None


def build_camera_error_message():
    """Return a macOS-friendly explanation when camera access fails."""
    if sys.platform == "darwin":
        return (
            "Camera access is blocked or unavailable.\n"
            "Open System Settings → Privacy & Security → Camera and allow access for "
            "Terminal, iTerm2, VS Code, or Python, then restart the app."
        )
    return "Camera access is blocked or unavailable."


def get_camera():
    """Get or initialize the camera. Thread-safe singleton with retry."""
    global camera
    with camera_lock:
        if camera is not None and camera.isOpened():
            return camera

        backend_candidates = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if sys.platform == "darwin" else [cv2.CAP_ANY]

        # Try index 0 first (built-in), then 1 (external/continuity)
        for idx in [0, 1]:
            for backend in backend_candidates:
                cam = cv2.VideoCapture(idx, backend)
                # Give the camera a moment to initialize
                time.sleep(0.3)
                if cam.isOpened():
                    # Verify we can actually read a frame
                    ret, _ = cam.read()
                    if ret:
                        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        camera = cam
                        print(f"✓ Camera opened on index {idx} using backend {backend}")
                        return camera
                    else:
                        cam.release()
                else:
                    cam.release()

        print("✗ Could not open any camera!")
        print(build_camera_error_message())
        camera = None
        return None


# ---------------------------------------------------------------------------
# Global state for recognition
# ---------------------------------------------------------------------------
# Always rebuild encodings from known_faces/ on startup so the pickle
# never goes stale (e.g. after the user deletes image folders).
from encode_faces import build_encodings as _initial_build
_initial_build()  # regenerates encodings.pickle from current known_faces/
known_encodings, known_names = load_encodings()
print(f"Loaded {len(set(known_names))} known person(s) with {len(known_encodings)} encoding(s)")

# Frame skipping: run recognition every N frames, reuse boxes on skipped frames
RECOGNIZE_EVERY_N = 3
frame_count = 0
last_boxes = []      # (top, right, bottom, left) in full-res coords
last_names = []      # matched names

# FPS tracking & timing
fps_value = 0.0
last_frame_time = time.time()

# Persistent face detection state: retains last successful detections across
# page navigations so predictions don't disappear.  Only updated when faces
# ARE actually detected — never auto-expires.
last_known_faces = []       # list of dicts: [{name, confidence, distance}, ...]
last_detection_time = 0.0   # timestamp of the most recent annotate_frame detection run

# Latest frame for enrollment capture
latest_frame = None
latest_frame_lock = threading.Lock()


def annotate_frame(frame):
    """Run face detection and draw overlays on a frame in place."""
    global frame_count, last_boxes, last_names, fps_value, last_frame_time

    if frame is None:
        return None

    frame_count += 1

    # --- Performance: downscale for detection, run every Nth frame ---
    if frame_count % RECOGNIZE_EVERY_N == 0:
        h, w = frame.shape[:2]
        scale = 320.0 / w
        small = cv2.resize(frame, (320, int(h * scale)))
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = detect_faces(rgb_small, model="hog")
        encodings = encode_faces(rgb_small, locations)
        names = match_faces(encodings, known_encodings, known_names)

        inv_scale = 1.0 / scale
        last_boxes = [
            (int(top * inv_scale), int(right * inv_scale),
             int(bottom * inv_scale), int(left * inv_scale))
            for (top, right, bottom, left) in locations
        ]
        last_names = names

        # Persist non-empty detections so they survive page navigations
        global last_known_faces, last_detection_time
        last_detection_time = time.time()
        face_list = []
        for match in names:
            if isinstance(match, dict):
                face_list.append({
                    "name": match.get("name", "Unknown"),
                    "confidence": match.get("confidence", 0),
                    "distance": match.get("distance", 1.0),
                })
            else:
                face_list.append({"name": str(match), "confidence": 0, "distance": 1.0})
        if face_list:
            last_known_faces = face_list

    # --- Draw boxes and names on the frame ---
    for (top, right, bottom, left), match in zip(last_boxes, last_names):
        if isinstance(match, dict):
            name = match.get("name", "Unknown")
            confidence = match.get("confidence", 0)
        else:
            name = str(match)
            confidence = 0

        if name == "Unknown":
            color = (0, 50, 240)  # Red-orange highlight for unrecognized
            label = "Can't Recognize — Please Enroll"
        else:
            color = (0, 220, 100)  # Vibrant green highlight for recognized
            label = f"{name} ({confidence}%)"

        # Draw rounded bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Label background box calculation
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        label_y = max(top - 8, label_height + 8)

        # Background rectangle for text contrast
        cv2.rectangle(
            frame,
            (left, label_y - label_height - 6),
            (left + label_width + 12, label_y + baseline),
            color,
            cv2.FILLED
        )
        # Text label
        cv2.putText(
            frame,
            label,
            (left + 6, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # --- FPS counter ---
    now = time.time()
    dt = now - last_frame_time
    last_frame_time = now
    if dt > 0:
        current_fps = 1.0 / dt
        fps_value = 0.85 * fps_value + 0.15 * current_fps if fps_value > 0 else current_fps

    fps_text = f"FPS: {fps_value:.1f}"
    cv2.putText(frame, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# MJPEG generator
# ---------------------------------------------------------------------------
def generate_frames():
    """Yield MJPEG frames with face recognition annotations."""
    global frame_count, last_boxes, last_names, fps_value, latest_frame

    # Retry camera access a few times — it may take a moment after permission
    cam = None
    for attempt in range(3):
        cam = get_camera()
        if cam is not None:
            break
        print(f"  Camera retry {attempt + 1}/3...")
        time.sleep(1.0)

    if cam is None:
        # Yield a single error frame with helpful instructions
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Camera not available", (100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        lines = build_camera_error_message().split("\n")
        for offset, line in enumerate(lines):
            cv2.putText(error_img, line, (40, 250 + offset * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        _, buf = cv2.imencode(".jpg", error_img)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        return

    while True:
        with camera_read_lock:
            success, frame = cam.read()
            
        if not success:
            # If camera is busy or buffering, wait a tiny bit and retry
            time.sleep(0.01)
            continue

        # Store latest frame for enrollment capture
        annotated_frame = annotate_frame(frame)
        with latest_frame_lock:
            latest_frame = annotated_frame.copy()

        # Encode and yield as MJPEG
        _, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Main page with live video feed."""
    num_people = len(set(known_names)) if known_names else 0
    current_fps = round(fps_value, 1) if fps_value > 0 else "—"
    return render_template(
        "index.html",
        num_people=num_people,
        fps=current_fps,
        threshold=MATCH_THRESHOLD,
        last_faces=last_known_faces
    )


@app.route("/video_feed")
def video_feed():
    """MJPEG streaming endpoint."""
    response = Response(generate_frames(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/video_frame")
def video_frame():
    """Return the latest processed camera frame as a JPEG for live UI updates."""
    with latest_frame_lock:
        if latest_frame is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for camera", (140, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(placeholder, "Allow camera access in System Settings", (70, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            _, buffer = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return Response(buffer.tobytes(), mimetype="image/jpeg")

        _, buffer = cv2.imencode(".jpg", latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    response = Response(buffer.tobytes(), mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    """Accept a JPEG frame from the browser and update the latest annotated frame."""
    try:
        uploaded = request.files.get("frame")
        if uploaded is None:
            return jsonify({"error": "No frame uploaded"}), 400

        if getattr(uploaded, "filename", "") == "":
            return jsonify({"error": "No frame uploaded"}), 400

        data = uploaded.read()
        if not data:
            return jsonify({"error": "Empty frame"}), 400

        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode frame"}), 400

        annotated_frame = annotate_frame(frame)
        if annotated_frame is None:
            return jsonify({"error": "Frame annotation failed"}), 400

        with latest_frame_lock:
            latest_frame = annotated_frame.copy()

        return jsonify({"success": True})
    except Exception as exc:
        return jsonify({"error": f"Upload failed: {exc}"}), 400


@app.route("/enroll", methods=["GET"])
def enroll_page():
    """Enrollment page — capture faces for a new person."""
    return render_template("enroll.html")


@app.route("/capture_frame", methods=["GET"])
def capture_frame():
    """Return the current camera frame as JPEG for enrollment or live refresh."""
    with latest_frame_lock:
        if latest_frame is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for camera", (140, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(placeholder, "Allow camera access in System Settings", (70, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            _, buffer = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return Response(buffer.tobytes(), mimetype="image/jpeg")
        _, buffer = cv2.imencode(".jpg", latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(buffer.tobytes(), mimetype="image/jpeg")


@app.route("/enroll", methods=["POST"])
def enroll_submit():
    """Receive captured frames and save to known_faces/<name>/."""
    global known_encodings, known_names

    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400

    files = request.files.getlist("images")
    if not files or len(files) == 0:
        return jsonify({"error": "No images provided"}), 400

    # Create the person's directory
    person_dir = os.path.join("known_faces", name)
    os.makedirs(person_dir, exist_ok=True)

    saved_count = 0
    for i, file in enumerate(files):
        if file and file.filename:
            filepath = os.path.join(person_dir, f"{name}_{i+1}.jpg")
            file.save(filepath)
            saved_count += 1

    if saved_count == 0:
        return jsonify({"error": "No valid images saved"}), 400

    # Re-build encodings without sys.exit
    from encode_faces import build_encodings
    success, message = build_encodings()

    if not success:
        return jsonify({"error": f"Encoding failed: {message}"}), 400

    # Reload into memory
    known_encodings, known_names = load_encodings()

    return jsonify({
        "success": True,
        "message": f"Enrolled '{name}' with {saved_count} photo(s). System updated!"
    })


@app.route("/status")
def status():
    """API endpoint for current status info and live detected faces."""
    now = time.time()
    time_since = now - last_detection_time if last_detection_time else 999
    is_camera_live = time_since < 2.0

    active_faces = []
    # If camera is actively detecting faces in the current frame, return them
    if is_camera_live and last_names:
        for match in last_names:
            if isinstance(match, dict):
                active_faces.append({
                    "name": match.get("name", "Unknown"),
                    "confidence": match.get("confidence", 0),
                    "distance": match.get("distance", 1.0),
                    "is_live": True,
                })
            else:
                active_faces.append({
                    "name": str(match), "confidence": 0,
                    "distance": 1.0, "is_live": True,
                })
    elif last_known_faces:
        # If camera has a gap between detections or user navigated back, retain last known faces
        for face in last_known_faces:
            active_faces.append({**face, "is_live": False})

    return jsonify({
        "num_people": len(set(known_names)) if known_names else 0,
        "num_encodings": len(known_encodings),
        "fps": round(fps_value, 1) if fps_value > 0 else "—",
        "threshold": MATCH_THRESHOLD,
        "detected_faces": active_faces,
        "camera_active": is_camera_live,
    })


@app.route("/people", methods=["GET"])
def people_page():
    """Page for viewing and managing enrolled people."""
    return render_template("people.html")


@app.route("/api/people", methods=["GET"])
def api_get_people():
    """Return JSON list of all enrolled people and their photos."""
    known_faces_dir = os.path.join(os.path.dirname(__file__), "known_faces")
    if not os.path.exists(known_faces_dir):
        return jsonify({"people": []})

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    people = []

    for item in sorted(os.listdir(known_faces_dir)):
        person_dir = os.path.join(known_faces_dir, item)
        if os.path.isdir(person_dir):
            photos = sorted([
                f for f in os.listdir(person_dir)
                if os.path.splitext(f)[1].lower() in image_extensions
            ])
            people.append({
                "name": item,
                "num_photos": len(photos),
                "photos": photos
            })

    return jsonify({"people": people})


@app.route("/api/people/<name>/photo/<filename>", methods=["GET"])
def api_get_person_photo(name, filename):
    """Serve a specific photo of an enrolled person."""
    known_faces_dir = os.path.join(os.path.dirname(__file__), "known_faces")
    person_dir = os.path.join(known_faces_dir, name)
    if not os.path.exists(person_dir):
        return jsonify({"error": "Person not found"}), 404
    return send_from_directory(person_dir, filename)


@app.route("/api/people/<name>", methods=["DELETE"])
def api_delete_person(name):
    """Delete an enrolled person and instantly update encodings in memory at runtime."""
    global known_encodings, known_names, last_boxes, last_names

    known_faces_dir = os.path.join(os.path.dirname(__file__), "known_faces")
    person_dir = os.path.join(known_faces_dir, name)

    if not os.path.exists(person_dir):
        return jsonify({"error": f"Person '{name}' not found"}), 404

    try:
        shutil.rmtree(person_dir)
    except Exception as exc:
        return jsonify({"error": f"Failed to delete directory: {exc}"}), 500

    # Rebuild encodings immediately
    from encode_faces import build_encodings
    build_encodings()

    # Reload encodings into memory state
    known_encodings, known_names = load_encodings()
    last_boxes = []
    last_names = []

    return jsonify({
        "success": True,
        "message": f"Successfully deleted '{name}'. Model updated in real-time!",
        "num_people": len(set(known_names)) if known_names else 0
    })


@app.route("/refresh", methods=["POST"])
def refresh_encodings():
    """Force-rebuild encodings from known_faces/ and reload into memory."""
    global known_encodings, known_names, last_boxes, last_names
    from encode_faces import build_encodings
    success, message = build_encodings()
    known_encodings, known_names = load_encodings()
    last_boxes = []
    last_names = []
    return jsonify({
        "success": True,
        "num_people": len(set(known_names)) if known_names else 0,
        "num_encodings": len(known_encodings),
        "message": message
    })




def start_server(preferred_port=5001, max_attempts=10):
    """Start the Flask server on the preferred port, falling back to nearby ports if needed."""
    ports_to_try = [preferred_port] + [preferred_port + i for i in range(1, max_attempts)]
    for port in ports_to_try:
        try:
            print(f"Starting server on port {port}...")
            server = make_server("0.0.0.0", port, app)
            print(f"\n🎥 Face Recognition Web App")
            print(f"  → Open http://localhost:{port} in your browser\n")
            server.serve_forever()
            return
        except OSError as exc:
            if "Address already in use" not in str(exc):
                raise
            print(f"Port {port} is busy; trying {port + 1}...")
            continue

    raise RuntimeError(f"Could not start the server after trying ports {ports_to_try}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    preferred_port = int(os.environ.get("PORT") or os.environ.get("FLASK_PORT") or "5001")
    start_server(preferred_port)
