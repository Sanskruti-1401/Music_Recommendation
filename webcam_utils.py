"""
webcam_utils.py
---------------
Handles webcam snapshot capture using Streamlit's st.camera_input widget.
Uses snapshot mode (not live stream) — most stable approach for beginners.
Returns a PIL Image ready for emotion detection.
"""

import io
import numpy as np
import streamlit as st
from PIL import Image
from typing import Optional


def capture_webcam_snapshot() -> Optional[Image.Image]:
    """
    Render a Streamlit camera input widget and return the captured snapshot
    as a PIL Image, or None if no snapshot has been taken yet.

    NOTE:
    - Uses st.camera_input (snapshot mode) — NOT live video stream.
    - This approach is stable, beginner-friendly, and works across OS/browsers.
    - The user clicks "Take Photo" in the widget, then the image is processed.
    """
    st.info(
        "📸 **Click 'Take photo'** in the camera widget below to capture your expression.",
        icon="ℹ️",
    )

    camera_image = st.camera_input(
        label="Capture your face",
        key="webcam_snapshot",
        help="Allow browser camera access, then click 'Take photo'.",
    )

    if camera_image is not None:
        try:
            img_bytes = camera_image.getvalue()
            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return pil_image
        except Exception as e:
            st.error(f"Could not process webcam image: {e}")
            return None

    return None
