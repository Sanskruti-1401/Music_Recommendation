"""
recommender.py
--------------
Song recommendation engine.
Reads songs.csv and filters by emotion, language, and genre.
Returns a randomized subset of matching songs.
"""

import os
import random
import pandas as pd


# Path to the songs dataset (relative to project root)
SONGS_CSV_PATH = os.path.join(os.path.dirname(__file__), "songs.csv")

# Expected columns in the CSV
REQUIRED_COLUMNS = {"song_name", "artist", "emotion", "language", "genre", "link"}


def load_songs(csv_path: str = SONGS_CSV_PATH) -> pd.DataFrame:
    """
    Load and validate the songs CSV file.

    Returns a cleaned DataFrame or raises a descriptive error.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"songs.csv not found at: {csv_path}\n"
            "Make sure songs.csv is in the project root directory."
        )

    df = pd.read_csv(csv_path)

    # Strip whitespace from all string columns
    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

    # Lowercase key filter columns for consistent matching
    df["emotion"] = df["emotion"].str.lower()
    df["language"] = df["language"].str.lower()
    df["genre"] = df["genre"].str.lower()

    # Check for required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"songs.csv is missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    return df


def get_recommendations(
    emotion: str,
    language: str = "all",
    genre: str = "all",
    n: int = 5,
    exclude_songs: list = None,
) -> dict:
    """
    Return song recommendations for the given emotion.

    Parameters
    ----------
    emotion      : detected emotion string (e.g. "happy")
    language     : "all", "english", or "hindi"
    genre        : "all" or specific genre name
    n            : number of songs to return
    exclude_songs: list of song names already shown (to avoid repeats)

    Returns
    -------
    dict with keys:
        - success (bool)
        - songs (list of dicts)     each dict = one song row
        - count (int)
        - error (str)
    """
    try:
        df = load_songs()
    except (FileNotFoundError, ValueError) as e:
        return {"success": False, "songs": [], "count": 0, "error": str(e)}

    if df.empty:
        return {
            "success": False,
            "songs": [],
            "count": 0,
            "error": "songs.csv is empty.",
        }

    # ---- Filter by emotion ----
    emotion = emotion.lower().strip()
    filtered = df[df["emotion"] == emotion].copy()

    if filtered.empty:
        return {
            "success": False,
            "songs": [],
            "count": 0,
            "error": f"No songs found for emotion: '{emotion}'. "
                     "Check that songs.csv has entries for this emotion.",
        }

    # ---- Filter by language ----
    if language and language.lower() != "all":
        lang_filtered = filtered[filtered["language"] == language.lower()]
        # Fall back to all languages if filter leaves nothing
        if not lang_filtered.empty:
            filtered = lang_filtered

    # ---- Filter by genre ----
    if genre and genre.lower() != "all":
        genre_filtered = filtered[filtered["genre"] == genre.lower()]
        if not genre_filtered.empty:
            filtered = genre_filtered

    # ---- Exclude already-shown songs ----
    if exclude_songs:
        exclude_lower = [s.lower() for s in exclude_songs]
        filtered = filtered[~filtered["song_name"].str.lower().isin(exclude_lower)]

    # If after exclusion we have nothing, reset (allow repeats)
    if filtered.empty:
        filtered = df[df["emotion"] == emotion].copy()

    # ---- Sample up to n songs randomly ----
    sample_size = min(n, len(filtered))
    sampled = filtered.sample(n=sample_size, random_state=None)  # random_state=None = truly random

    songs_list = sampled.to_dict(orient="records")

    return {
        "success": True,
        "songs": songs_list,
        "count": len(songs_list),
        "error": None,
    }


def get_available_languages(csv_path: str = SONGS_CSV_PATH) -> list:
    """Return sorted list of unique languages in the dataset."""
    try:
        df = load_songs(csv_path)
        langs = sorted(df["language"].dropna().unique().tolist())
        return ["All"] + [l.capitalize() for l in langs]
    except Exception:
        return ["All", "English", "Hindi"]


def get_available_genres(csv_path: str = SONGS_CSV_PATH) -> list:
    """Return sorted list of unique genres in the dataset."""
    try:
        df = load_songs(csv_path)
        genres = sorted(df["genre"].dropna().unique().tolist())
        return ["All"] + [g.title() for g in genres]
    except Exception:
        return ["All"]