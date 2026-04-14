"""
app.py
------
Main Streamlit application for the Emotion-Based Music Recommendation System.

Run with:
    streamlit run app.py
"""

import streamlit as st
from PIL import Image

# ── Local module imports ──
from emotion_detector import detect_emotion, get_emotion_info
from recommender import get_recommendations, get_available_languages, get_available_genres
from webcam_utils import capture_webcam_snapshot
from utils import (
    init_session_history,
    add_to_history,
    get_shown_songs,
    clear_history,
    get_emotion_history_counts,
    validate_image,
    make_emotion_bar_chart,
    make_session_history_chart,
    format_song_table,
    confidence_badge_html,
)


# ══════════════════════════════════════════════
#  Page Configuration
# ══════════════════════════════════════════════

st.set_page_config(
    page_title="🎵 Emotion Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize session state ──
init_session_history()


# ══════════════════════════════════════════════
#  Custom CSS — Polished Dark UI
# ══════════════════════════════════════════════

st.markdown(
    """
    <style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* ── Header Banner ── */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF6B6B, #FFA07A, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-subtitle {
        color: #AAAAAA;
        font-size: 1.05rem;
        margin-top: 4px;
    }

    /* ── Emotion Card ── */
    .emotion-card {
        background: linear-gradient(135deg, #1A1D27, #252836);
        border-radius: 16px;
        padding: 28px 24px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        margin: 12px 0;
    }
    .emotion-emoji {
        font-size: 4rem;
        line-height: 1.2;
    }
    .emotion-label {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 8px;
    }

    /* ── Song Card ── */
    .song-card {
        background: #1A1D27;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 8px 0;
        border-left: 4px solid #FF6B6B;
        transition: transform 0.2s;
    }
    .song-card:hover {
        transform: translateX(4px);
    }
    .song-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #FFFFFF;
    }
    .song-artist {
        font-size: 0.88rem;
        color: #AAAAAA;
    }
    .song-meta {
        font-size: 0.78rem;
        color: #888888;
        margin-top: 4px;
    }
    .song-link a {
        color: #FF6B6B;
        font-size: 0.85rem;
        text-decoration: none;
        font-weight: 600;
    }

    /* ── Section Header ── */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 20px 0 10px 0;
        padding-bottom: 6px;
        border-bottom: 2px solid #FF6B6B;
    }

    /* ── Info chip ── */
    .chip {
        display: inline-block;
        background: rgba(255,107,107,0.15);
        border: 1px solid #FF6B6B;
        color: #FF6B6B;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        margin: 2px;
    }

    /* ── Divider ── */
    hr { border-color: rgba(255,255,255,0.08); }

    /* ── Sidebar ── */
    .css-1d391kg { background: #13151f; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════
#  Header
# ══════════════════════════════════════════════

st.markdown('<div class="hero-title">🎵 Emotion Music Recommender</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">AI-powered music recommendations based on your facial expression · '
    'Final Year Project · Computer Vision + Deep Learning</div>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ══════════════════════════════════════════════
#  Sidebar — Controls & Filters
# ══════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    # Input mode
    input_mode = st.radio(
        "📥 Input Mode",
        ["📤 Upload Image", "📷 Use Webcam"],
        index=0,
        help="Choose how to provide your face image.",
    )

    st.markdown("---")
    st.markdown("### 🎛️ Recommendation Filters")

    # Language filter
    available_languages = get_available_languages()
    selected_language = st.selectbox("🌐 Language", available_languages, index=0)

    # Genre filter
    available_genres = get_available_genres()
    selected_genre = st.selectbox("🎸 Genre", available_genres, index=0)

    # Number of recommendations
    num_recommendations = st.slider(
        "🎵 Number of Songs", min_value=3, max_value=10, value=5, step=1
    )

    st.markdown("---")

    # Refresh recommendations button (only if emotion detected)
    if st.session_state.get("current_emotion"):
        if st.button("🔄 Refresh Recommendations", use_container_width=True):
            st.session_state["refresh"] = True

    # Clear history
    if st.button("🗑️ Clear Session History", use_container_width=True):
        clear_history()
        st.success("Session history cleared!")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "Built with **DeepFace**, **OpenCV**, **Streamlit**. "
        "Detects 7 emotions: Happy, Sad, Angry, Fear, Surprise, Neutral, Disgust."
    )


# ══════════════════════════════════════════════
#  Main Layout — Two Columns
# ══════════════════════════════════════════════

col_input, col_result = st.columns([1, 1], gap="large")

# ── Left Column: Image Input ──
with col_input:
    st.markdown('<div class="section-header">📸 Capture Your Expression</div>', unsafe_allow_html=True)

    image_to_analyze = None

    if "📤 Upload Image" in input_mode:
        uploaded_file = st.file_uploader(
            "Upload a clear face photo",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a front-facing photo with your face visible.",
        )

        if uploaded_file:
            is_valid, result = validate_image(uploaded_file)
            if is_valid:
                image_to_analyze = result
                st.image(
                    image_to_analyze,
                    caption="Uploaded Image",
                    use_container_width=True,
                )
            else:
                st.error(f"❌ {result}")

    else:
        # Webcam snapshot mode
        webcam_image = capture_webcam_snapshot()
        if webcam_image:
            image_to_analyze = webcam_image
            st.image(
                webcam_image,
                caption="Webcam Snapshot",
                use_container_width=True,
            )

    # ── Analyze Button ──
    if image_to_analyze is not None:
        st.markdown("")
        analyze_clicked = st.button(
            "🧠 Analyze Emotion & Get Songs",
            type="primary",
            use_container_width=True,
        )
    else:
        analyze_clicked = False
        st.info("👆 Upload an image or take a webcam snapshot to get started.", icon="💡")


# ── Right Column: Results ──
with col_result:
    st.markdown('<div class="section-header">🎯 Detection Result</div>', unsafe_allow_html=True)

    # ── Run Detection ──
    if analyze_clicked and image_to_analyze is not None:
        with st.spinner("🔍 Analyzing your emotion..."):
            detection = detect_emotion(image_to_analyze)

        if detection["success"]:
            emotion = detection["emotion"]
            confidence = detection["confidence"]
            all_scores = detection["all_scores"]
            emotion_info = get_emotion_info(emotion)

            # Save to session
            st.session_state["current_emotion"] = emotion
            st.session_state["current_confidence"] = confidence
            st.session_state["current_all_scores"] = all_scores
            st.session_state["last_image"] = image_to_analyze

            # Show emotion card
            color = emotion_info["color"]
            emoji = emotion_info["emoji"]
            st.markdown(
                f"""
                <div class="emotion-card">
                    <div class="emotion-emoji">{emoji}</div>
                    <div class="emotion-label" style="color:{color};">{emotion.upper()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Confidence badge
            st.markdown(
                confidence_badge_html(confidence, color),
                unsafe_allow_html=True,
            )

            st.markdown("")

            # Emotion scores bar chart
            chart = make_emotion_bar_chart(all_scores)
            if chart:
                st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
            else:
                # Fallback: show raw scores
                st.write("**All emotion scores:**")
                for em, sc in sorted(all_scores.items(), key=lambda x: -x[1]):
                    st.write(f"- {em.capitalize()}: {sc:.1f}%")

        else:
            # Detection failed — show friendly message
            st.error(f"❌ {detection['error']}")
            if not detection["face_detected"]:
                st.warning(
                    "**Tips for better detection:**\n"
                    "- Ensure your face is clearly visible\n"
                    "- Use good lighting\n"
                    "- Face the camera directly\n"
                    "- Avoid sunglasses or face coverings",
                    icon="💡",
                )

    elif st.session_state.get("current_emotion"):
        # Persist last result without re-running detection
        emotion = st.session_state["current_emotion"]
        confidence = st.session_state.get("current_confidence", 0)
        all_scores = st.session_state.get("current_all_scores", {})
        emotion_info = get_emotion_info(emotion)
        color = emotion_info["color"]
        emoji = emotion_info["emoji"]

        st.markdown(
            f"""
            <div class="emotion-card">
                <div class="emotion-emoji">{emoji}</div>
                <div class="emotion-label" style="color:{color};">{emotion.upper()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(confidence_badge_html(confidence, color), unsafe_allow_html=True)

        chart = make_emotion_bar_chart(all_scores)
        if chart:
            st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})

    else:
        st.markdown(
            "<br><br><div style='text-align:center; color:#555; font-size:1.1rem;'>"
            "Your emotion result will appear here after analysis.</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════
#  Music Recommendations Section
# ══════════════════════════════════════════════

st.markdown("---")
st.markdown('<div class="section-header">🎧 Recommended Songs</div>', unsafe_allow_html=True)

current_emotion = st.session_state.get("current_emotion")

# Trigger recommendations on: analyze, refresh button, or page load with existing emotion
_refresh = st.session_state.pop("refresh", False)
show_recommendations = (
    (analyze_clicked and current_emotion)
    or _refresh
)

if show_recommendations and current_emotion:
    lang_filter = selected_language if selected_language != "All" else "all"
    genre_filter = selected_genre if selected_genre != "All" else "all"

    reco = get_recommendations(
        emotion=current_emotion,
        language=lang_filter,
        genre=genre_filter,
        n=num_recommendations,
        exclude_songs=get_shown_songs(),
    )

    if reco["success"]:
        add_to_history(current_emotion, reco["songs"])

        emotion_info = get_emotion_info(current_emotion)
        st.success(
            f"Found **{reco['count']} songs** for your mood: "
            f"{emotion_info['emoji']} **{current_emotion.upper()}**"
        )

        # Display as styled cards
        for song in reco["songs"]:
            lang_chip = f'<span class="chip">🌐 {song.get("language","").capitalize()}</span>'
            genre_chip = f'<span class="chip">🎸 {song.get("genre","").title()}</span>'
            yt_link = song.get("link", "")
            yt_html = (
                f'<div class="song-link"><a href="{yt_link}" target="_blank">▶ Watch on YouTube</a></div>'
                if yt_link
                else ""
            )
            st.markdown(
                f"""
                <div class="song-card">
                    <div class="song-title">🎵 {song.get('song_name', '—')}</div>
                    <div class="song-artist">🎤 {song.get('artist', '—')}</div>
                    <div class="song-meta">{lang_chip} {genre_chip}</div>
                    {yt_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Spacer
        st.markdown("<br>", unsafe_allow_html=True)

        # Show as table option
        with st.expander("📋 View as Table"):
            df_table = format_song_table(reco["songs"])
            if not df_table.empty:
                st.dataframe(df_table, use_container_width=True, hide_index=True)

    else:
        st.warning(f"⚠️ {reco['error']}", icon="🎵")

elif current_emotion and not show_recommendations:
    st.info(
        f"Click **'Analyze Emotion & Get Songs'** again or use **'🔄 Refresh Recommendations'** "
        f"in the sidebar to get new songs for your current mood.",
        icon="💡",
    )
else:
    st.markdown(
        "<div style='text-align:center; color:#555; padding: 30px;'>"
        "🎵 Music recommendations will appear here after emotion detection.</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
#  Session History Section
# ══════════════════════════════════════════════

history = st.session_state.get("history", [])
emotion_counts = get_emotion_history_counts()

if history:
    st.markdown("---")
    col_hist, col_pie = st.columns([1, 1], gap="large")

    with col_hist:
        st.markdown('<div class="section-header">🕐 Detection History</div>', unsafe_allow_html=True)
        for entry in reversed(history[-8:]):  # Show last 8 entries
            info = get_emotion_info(entry["emotion"])
            songs_preview = ", ".join(entry["songs"][:3])
            if len(entry["songs"]) > 3:
                songs_preview += f" +{len(entry['songs'])-3} more"
            st.markdown(
                f"**{entry['timestamp']}** — {info['emoji']} **{entry['emotion'].upper()}** "
                f"· _{songs_preview}_"
            )

    with col_pie:
        st.markdown('<div class="section-header">📊 Session Emotion Distribution</div>', unsafe_allow_html=True)
        pie_chart = make_session_history_chart(emotion_counts)
        if pie_chart:
            st.plotly_chart(pie_chart, use_container_width=True, config={"displayModeBar": False})
        else:
            for emotion, count in emotion_counts.items():
                info = get_emotion_info(emotion)
                st.write(f"{info['emoji']} {emotion.capitalize()}: {count} time(s)")


# ══════════════════════════════════════════════
#  Footer
# ══════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#555; font-size:0.8rem; padding:10px;'>"
    "Emotion-Based Music Recommendation System · Final Year Project · "
    "Built with ❤️ using DeepFace, OpenCV & Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
