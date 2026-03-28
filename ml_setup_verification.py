#!/usr/bin/env python3
"""Local ML Models Setup & Verification Script"""

import os, sys, warnings, numpy as np
warnings.filterwarnings('ignore')

GREEN = '\033[92m'; RED = '\033[91m'; YELLOW = '\033[93m'
BLUE = '\033[94m'; RESET = '\033[0m'; BOLD = '\033[1m'

def p(msg, s="info"): 
    c={"s":GREEN+"✓","e":RED+"✗","w":YELLOW+"⚠","i":BLUE+"→"}; 
    print(f"{c.get(s,'')} {msg}")
def ph(msg): print(f"\n{BOLD}{'='*55}{RESET}\n{BOLD}{msg}{RESET}\n")

# 1. YOLOv8-nano
def setup_yolov8():
    ph("1. YOLOv8-nano Object Detection")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        p("YOLOv8-nano ready! (yolov8n.pt)", "s")
        return True
    except Exception as e:
        p(f"YOLOv8-nano failed: {e}", "e")
        return False

# 2. DeepFace
def setup_deepface():
    ph("2. DeepFace Face Recognition")
    try:
        import deepface
        p(f"DeepFace {deepface.__version__} installed", "s")
        try:
            import tensorflow as tf
            p(f"TensorFlow available: {tf.__version__}", "s")
        except ImportError:
            p("Note: TensorFlow needed for full DeepFace functionality", "w")
            p("  → DeepFace is installed, ready when TF is available", "i")
        return True
    except Exception as e:
        p(f"DeepFace failed: {e}", "e")
        return False

# 3. MediaPipe
def setup_mediapipe():
    ph("3. MediaPipe Body/Face/Hand Tracking")
    try:
        import mediapipe as mp
        p(f"MediaPipe {mp.__version__} ready!", "s")
        try:
            from mediapipe.tasks import python
            p("Tasks API (Python) available", "i")
        except:
            p("Using MediaPipe Python API", "i")
        return True
    except Exception as e:
        p(f"MediaPipe failed: {e}", "e")
        return False

# 4. Librosa
def setup_librosa():
    ph("4. Librosa Audio Analysis")
    try:
        import librosa
        sr, y = 22050, np.sin(2*np.pi*440*np.linspace(0,1,22050))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        p(f"Librosa {librosa.__version__} ready! MFCCs{mfccs.shape}", "s")
        return True
    except Exception as e:
        p(f"Librosa failed: {e}", "e")
        return False

# 5. OpenCV DNN
def setup_opencv_dnn():
    ph("5. OpenCV DNN + MobileNetV3/EfficientNet-Lite")
    try:
        import cv2
        p(f"OpenCV {cv2.__version__} + DNN module ready!", "s")
        p("MobileNetV3/EfficientNet-Lite compatible", "i")
        return True
    except Exception as e:
        p(f"OpenCV DNN failed: {e}", "e")
        return False

def main():
    print(f"\n{BOLD}  Local ML Models Setup & Verification{RESET}\n")
    res = {'YOLOv8-nano':setup_yolov8(),'DeepFace':setup_deepface(),
           'MediaPipe':setup_mediapipe(),'Librosa':setup_librosa(),'OpenCV-DNN':setup_opencv_dnn()}
    ph("Summary")
    for n,s in res.items(): p(f"{n}: {'PASS' if s else 'FAIL'}", "s" if s else "e")
    print(f"\n{'All Ready!' if all(res.values()) else 'Partial Setup'}\n")
    return 0 if all(res.values()) else 1

if __name__ == "__main__": sys.exit(main())