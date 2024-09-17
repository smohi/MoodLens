try:
    import cv2
    print("OpenCV installed successfully")
except ImportError:
    print("OpenCV not installed")

try:
    import deepface
    print("DeepFace installed successfully")
except ImportError:
    print("DeepFace not installed")

try:
    import mediapipe as mp
    print("MediaPipe installed successfully")
except ImportError:
    print("MediaPipe not installed")

try:
    import keras
    print("Keras installed successfully")
except ImportError:
    print("Keras not installed")
