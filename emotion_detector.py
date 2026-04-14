"""
emotion_detector.py
--------------------
Handles face detection and emotion classification.
Uses DeepFace as the primary backend (with fallback messaging).
Returns dominant emotion, confidence score, and all emotion scores.
"""

import numpy as np
import cv2
from PIL import Image


# Mapping DeepFace emotion labels to our standard set (they already match, but kept for safety)
EMOTION_LABELS = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fear",
    "surprise": "surprise",
    "neutral": "neutral",
    "disgust": "disgust",
}

# Emoji map for UI display
EMOTION_EMOJI = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "surprise": "😮",
    "neutral": "😐",
    "disgust": "🤢",
}

# Color map for emotion badges
EMOTION_COLOR = {
    "happy": "#FFD700",
    "sad": "#4A90D9",
    "angry": "#E74C3C",
    "fear": "#8E44AD",
    "surprise": "#F39C12",
    "neutral": "#95A5A6",
    "disgust": "#27AE60",
}


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a BGR OpenCV numpy array."""
    rgb_array = np.array(pil_image.convert("RGB"))
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert a BGR OpenCV numpy array to a PIL Image."""
    rgb_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_array)


def detect_emotion(image_input) -> dict:
    """
    Main emotion detection function.

    Parameters
    ----------
    image_input : PIL.Image or np.ndarray
        The input image (from upload or webcam snapshot).

    Returns
    -------
    dict with keys:
        - success (bool)
        - emotion (str)          dominant emotion label
        - confidence (float)     0–100 confidence %
        - all_scores (dict)      all emotion probabilities
        - error (str)            error message if success=False
        - face_detected (bool)
    """
    try:
        # ---- Import DeepFace (lazy import to avoid crash at module load) ----
        from deepface import DeepFace

        # Convert PIL to numpy array if needed
        if isinstance(image_input, Image.Image):
            img_array = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            # If BGR (from OpenCV), convert to RGB for DeepFace
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                img_array = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            else:
                img_array = image_input
        else:
            return {
                "success": False,
                "error": "Unsupported image format.",
                "face_detected": False,
            }

        # ---- Run DeepFace emotion analysis ----
        result = DeepFace.analyze(
            img_path=img_array,
            actions=["emotion"],
            enforce_detection=True,   # Raises error if no face
            detector_backend="opencv",  # Fast and stable
            silent=True,
        )

        # DeepFace returns a list when multiple faces detected; take the first
        if isinstance(result, list):
            result = result[0]

        raw_emotions: dict = result.get("emotion", {})
        dominant_emotion: str = result.get("dominant_emotion", "neutral")

        # Normalize emotion label
        dominant_emotion = EMOTION_LABELS.get(dominant_emotion.lower(), "neutral")

        # Build all_scores with normalized keys
        all_scores = {
            EMOTION_LABELS.get(k.lower(), k.lower()): round(float(v), 2)
            for k, v in raw_emotions.items()
        }

        # Confidence = score of the dominant emotion
        confidence = all_scores.get(dominant_emotion, 0.0)

        return {
            "success": True,
            "emotion": dominant_emotion,
            "confidence": round(confidence, 2),
            "all_scores": all_scores,
            "face_detected": True,
            "error": None,
        }

    except ValueError as ve:
        # DeepFace raises ValueError when no face is detected
        err_str = str(ve).lower()
        if "face" in err_str or "detector" in err_str or "no face" in err_str:
            return {
                "success": False,
                "error": "No face detected in the image. Please try again with a clearer face photo.",
                "face_detected": False,
            }
        return {
            "success": False,
            "error": f"Detection error: {ve}",
            "face_detected": False,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error during emotion detection: {e}",
            "face_detected": False,
        }


def get_emotion_info(emotion: str) -> dict:
    """Return display info (emoji, color) for a given emotion label."""
    emotion = emotion.lower()
    return {
        "label": emotion,
        "emoji": EMOTION_EMOJI.get(emotion, "🎵"),
        "color": EMOTION_COLOR.get(emotion, "#AAAAAA"),
    }
