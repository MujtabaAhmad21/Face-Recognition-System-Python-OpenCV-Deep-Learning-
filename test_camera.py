"""
test_camera.py — Quick camera permission test.

Run this FIRST to trigger the macOS camera permission prompt.
Once your terminal app has camera access, app.py will work.

Usage:
    python test_camera.py
"""

import os
import sys

# Skip OpenCV's own auth request — we want to detect the permission state
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

import cv2


def test_camera():
    print("Testing camera access...\n")

    for idx in [0, 1]:
        print(f"  Trying camera index {idx}...", end=" ")
        cap = cv2.VideoCapture(idx)

        if not cap.isOpened():
            print("not available")
            cap.release()
            continue

        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"✓ Working! ({w}x{h})")
            print(f"\n✅ Camera is ready. You can now run: python app.py")
            return True
        else:
            print("opened but can't read frames (permission denied?)")

    # If we get here, no camera worked
    print("\n" + "=" * 60)
    print("  ✗ Camera access DENIED or no camera found.")
    print("=" * 60)
    print()
    print("  To fix this, grant camera access to your terminal app:")
    print()
    print("  1. Open System Settings")
    print("  2. Go to Privacy & Security → Camera")
    print("  3. Find your terminal app and enable it:")

    # Try to detect which terminal is running this
    term = os.environ.get("TERM_PROGRAM", "")
    if "iTerm" in term:
        print("     → Enable 'iTerm2'")
    elif "Apple_Terminal" in term:
        print("     → Enable 'Terminal'")
    elif "vscode" in term.lower():
        print("     → Enable 'Visual Studio Code'")
    else:
        print(f"     → Look for your terminal app (detected: {term or 'unknown'})")

    print("  4. You may need to restart the terminal after granting access")
    print("  5. Then run this script again: python test_camera.py")
    print()
    return False


if __name__ == "__main__":
    success = test_camera()
    sys.exit(0 if success else 1)
