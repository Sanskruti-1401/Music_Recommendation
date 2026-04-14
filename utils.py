"""
utils.py
--------
Shared utility functions used across the project.
Includes: session history, chart generation, image validation.
"""

import io
import base64
import datetime
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image


# ──────────────────────────────────────────────
#  Session History
# ──────────────────────────────────────────────

def init_session_history():
    """Initialize recommendation history in Streamlit session state."""
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "shown_songs" not in st.session_state:
        st.session_state["shown_songs"] = []
    if "current_emotion" not in st.session_state:
        st.session_state["current_emotion"] = None


def add_to_history(emotion: str, songs: list):
    """Append a detection result to session history."""
    init_session_history()
    entry = {
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "emotion": emotion,
        "songs": [s["song_name"] for s in songs],
    }
    st.session_state["history"].append(entry)
    # Track shown songs to avoid duplicates
    st.session_state["shown_songs"].extend([s["song_name"] for s in songs])


def get_shown_songs() -> list:
    """Return list of song names already shown this session."""
    init_session_history()
    return st.session_state.get("shown_songs", [])


def clear_history():
    """Reset session history and shown songs."""
    st.session_state["history"] = []
    st.session_state["shown_songs"] = []
    st.session_state["current_emotion"] = None


def get_emotion_history_counts() -> dict:
    """Return a dict of {emotion: count} from session history."""
    init_session_history()
    counts = {}
    for entry in st.session_state["history"]:
        emotion = entry["emotion"]
        counts[emotion] = counts.get(emotion, 0) + 1
    return counts


# ──────────────────────────────────────────────
#  Image Validation
# ──────────────────────────────────────────────

def validate_image(uploaded_file) -> tuple:
    """
    Validate an uploaded image file.

    Returns (is_valid: bool, image_or_error: PIL.Image | str)
    """
    if uploaded_file is None:
        return False, "No file uploaded."

    try:
        img = Image.open(uploaded_file)
        img.verify()  # Checks for corruption
        # Re-open after verify (verify closes the file)
        uploaded_file.seek(0)
        img = Image.open(uploaded_file).convert("RGB")

        # Basic size check
        w, h = img.size
        if w < 50 or h < 50:
            return False, "Image is too small. Please upload a clear face photo."
        if w > 4000 or h > 4000:
            # Resize large images for faster processing
            img = img.resize((1024, int(1024 * h / w)), Image.Resampling.LANCZOS)

        return True, img

    except Exception as e:
        return False, f"Invalid image file: {e}"


def pil_image_to_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    """Convert a PIL image to bytes (for display in Streamlit)."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


# ──────────────────────────────────────────────
#  Chart Generation (Plotly)
# ──────────────────────────────────────────────

def make_emotion_bar_chart(all_scores: dict):
    """
    Build a Plotly horizontal bar chart of emotion confidence scores.
    Returns a plotly Figure object.
    """
    try:
        import plotly.graph_objects as go
        from emotion_detector import EMOTION_COLOR, EMOTION_EMOJI

        emotions = list(all_scores.keys())
        scores = [all_scores[e] for e in emotions]
        colors = [EMOTION_COLOR.get(e, "#AAAAAA") for e in emotions]
        labels = [f"{EMOTION_EMOJI.get(e,'🎵')} {e.capitalize()}" for e in emotions]

        # Sort by score descending
        sorted_pairs = sorted(zip(scores, labels, colors), reverse=True)
        scores_sorted, labels_sorted, colors_sorted = zip(*sorted_pairs) if sorted_pairs else ([], [], [])

        fig = go.Figure(
            go.Bar(
                x=list(scores_sorted),
                y=list(labels_sorted),
                orientation="h",
                marker_color=list(colors_sorted),
                text=[f"{s:.1f}%" for s in scores_sorted],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Emotion Confidence Scores",
            xaxis_title="Confidence (%)",
            yaxis_title="",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EEEEEE"),
            height=300,
            margin=dict(l=10, r=60, t=40, b=10),
            xaxis=dict(range=[0, 110]),
        )
        return fig

    except ImportError:
        return None


def make_session_history_chart(emotion_counts: dict):
    """
    Build a Plotly pie chart of emotion detection history for this session.
    Returns a plotly Figure object, or None if plotly not installed.
    """
    try:
        import plotly.graph_objects as go
        from emotion_detector import EMOTION_COLOR, EMOTION_EMOJI

        if not emotion_counts:
            return None

        labels = [f"{EMOTION_EMOJI.get(e,'🎵')} {e.capitalize()}" for e in emotion_counts]
        values = list(emotion_counts.values())
        colors = [EMOTION_COLOR.get(e, "#AAAAAA") for e in emotion_counts]

        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                hole=0.4,
            )
        )
        fig.update_layout(
            title="Session Emotion Distribution",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EEEEEE"),
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=True,
        )
        return fig

    except ImportError:
        return None


# ──────────────────────────────────────────────
#  Formatting Helpers
# ──────────────────────────────────────────────

def format_song_table(songs: list) -> pd.DataFrame:
    """
    Convert list of song dicts to a display-ready DataFrame.
    Adds a clickable YouTube link column.
    """
    if not songs:
        return pd.DataFrame()

    rows = []
    for s in songs:
        rows.append({
            "🎵 Song": s.get("song_name", "—"),
            "🎤 Artist": s.get("artist", "—"),
            "🌐 Language": s.get("language", "—").capitalize(),
            "🎸 Genre": s.get("genre", "—").title(),
            "▶️ YouTube": s.get("link", ""),
        })
    return pd.DataFrame(rows)


def confidence_badge_html(confidence: float, color: str) -> str:
    """Return an HTML badge string for the confidence percentage."""
    return (
        f'<div style="display:inline-block; background:{color}; color:#000; '
        f'padding:6px 16px; border-radius:20px; font-weight:700; font-size:1.1rem;">'
        f"Confidence: {confidence:.1f}%</div>"
    )