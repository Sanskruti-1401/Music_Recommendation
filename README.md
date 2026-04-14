# 🎵 Emotion-Based Music Recommendation System Using Computer Vision

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)
![DeepFace](https://img.shields.io/badge/DeepFace-0.0.93%2B-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-lightblue?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **Final Year Project** — An end-to-end AI-powered web application that detects your facial emotion in real-time and recommends music that matches your current mood.

---

## 📌 Overview

This project uses **computer vision** and **deep learning** to:
1. Capture your face via webcam or uploaded image
2. Detect your current emotion using the **DeepFace** library
3. Recommend songs from a curated dataset that match your mood
4. Display everything in a clean, interactive **Streamlit** web app

**No model training required.** The system uses pre-trained models from DeepFace (built on FaceNet/OpenCV), making it easy to run on any laptop.

---

## ✨ Features

- 😄 **7 Emotion Classes**: Happy, Sad, Angry, Fear, Surprise, Neutral, Disgust
- 📷 **Dual Input Modes**: Upload image or use webcam snapshot
- 🎵 **Smart Song Matching**: Songs filtered by detected emotion, language, and genre
- 📊 **Confidence Visualization**: Plotly bar chart showing all emotion scores
- 🔄 **Refresh Recommendations**: Get new songs for the same emotion
- 🕐 **Session History**: Tracks all detections during your session
- 🌐 **Language Filter**: English, Hindi, or All
- 🎸 **Genre Filter**: Pop, Rock, Bollywood, Classical, and more
- ▶️ **YouTube Links**: One-click song playback
- 🌙 **Dark Theme UI**: Professional dark-mode Streamlit interface

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend / UI | Streamlit |
| Emotion Detection | DeepFace (pre-trained CNN) |
| Face Detection | OpenCV (Haar Cascade) |
| Image Processing | Pillow, NumPy |
| Data Handling | Pandas |
| Visualization | Plotly |
| Song Dataset | Custom songs.csv (65+ songs) |

---

## 📂 Folder Structure

```
emotion_music_recommender/
│
├── app.py                  ← Main Streamlit application
├── emotion_detector.py     ← Face & emotion detection logic
├── recommender.py          ← Song recommendation engine
├── utils.py                ← Helper functions (charts, history, validation)
├── webcam_utils.py         ← Webcam snapshot handler
├── songs.csv               ← Music dataset (65+ songs with YouTube links)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── .streamlit/
│   └── config.toml         ← Dark theme Streamlit config
│
├── assets/
│   └── sample_images/      ← Sample test images (add your own)
│
└── outputs/
    └── logs/               ← (Optional) Log files
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or 3.11 (recommended)
- Webcam (optional, for live capture)
- ~2 GB disk space (for TensorFlow + model weights)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/emotion-music-recommender.git
cd emotion-music-recommender
```

### Step 2: Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ First-time installation may take 5–10 minutes (TensorFlow is large).
> DeepFace model weights (~100MB) download automatically on first run.

### Step 4: Run the App
```bash
streamlit run app.py
```
Then open your browser at **http://localhost:8501**

---

## 📸 Screenshots

| Emotion Detection | Song Recommendations |
|:-----------------:|:--------------------:|
| *(Add screenshot here)* | *(Add screenshot here)* |

---

## 🎯 How It Works

```
User Face Image
      │
      ▼
Face Detection (OpenCV)
      │
      ▼
Emotion Classification (DeepFace CNN)
      │
      ▼
Dominant Emotion + Confidence Scores
      │
      ▼
Filter songs.csv by emotion + language + genre
      │
      ▼
Random sample of N matching songs
      │
      ▼
Display in Streamlit UI with YouTube links
```

---

## ⚠️ Known Limitations

- DeepFace requires a clear, front-facing photo with good lighting
- Performance depends on lighting conditions and image quality
- Webcam snapshot mode (not live stream) is used for stability
- TensorFlow is a heavy dependency (~500MB+)
- May be slower on CPU-only machines (use a laptop with decent RAM)

---

## 🔮 Future Scope

- [ ] Spotify API integration for real playlist generation
- [ ] Real-time video stream emotion tracking
- [ ] User accounts with persistent history (SQLite)
- [ ] More songs with genre/mood tagging
- [ ] Mobile-friendly PWA version
- [ ] Age/gender-based recommendation personalization
- [ ] Multi-face detection support

---

## 📄 Resume Description

> **Emotion-Based Music Recommendation System** | Python, DeepFace, OpenCV, Streamlit
> - Built an end-to-end AI web app that detects facial emotions using DeepFace (pre-trained CNN) and recommends contextually matching songs.
> - Implemented modular architecture separating detection, recommendation, and UI layers.
> - Integrated OpenCV for face detection, Pandas for CSV-based recommendation filtering, and Plotly for interactive confidence visualization.
> - Deployed as a Streamlit web application with webcam integration, session history tracking, and YouTube link support.

---

## 🎓 Viva / Interview Preparation

### Q1: What is DeepFace and why did you use it?
**A:** DeepFace is an open-source Python library by Facebook that wraps multiple pre-trained facial analysis models (VGG-Face, FaceNet, OpenFace, DeepFace). I used it because it provides accurate emotion detection without requiring me to train a model from scratch, which makes it practical for a project of this scope.

### Q2: How does emotion detection work in your project?
**A:** The input image is first passed through OpenCV's Haar Cascade face detector to locate the face region. DeepFace then runs this cropped face region through a pre-trained CNN that was originally trained on the FER2013 dataset containing ~35,000 labeled facial expression images. The network outputs probability scores for 7 emotion classes, and the highest-scoring one becomes the dominant emotion.

### Q3: Why Streamlit over Flask/Django?
**A:** Streamlit is a Python-native framework specifically designed for data science and ML demos. It requires no HTML/CSS/JavaScript knowledge, integrates natively with Python objects like Pandas DataFrames and Plotly charts, and has a built-in camera input widget. This made it the fastest path to a functional, professional-looking demo.

### Q4: How does your recommendation engine work?
**A:** I use a CSV-based dataset of 65+ songs, each tagged with emotion, language, and genre metadata. When an emotion is detected, I filter the DataFrame for matching rows, apply optional language/genre filters using Pandas boolean indexing, and randomly sample N songs. This avoids recommendation bias and is easy to extend with more songs.

### Q5: How do you handle errors — what if no face is detected?
**A:** DeepFace raises a `ValueError` when it cannot find a face in the image. I catch this exception in `emotion_detector.py` and return a structured error response instead of crashing. The Streamlit app then displays a user-friendly message with tips for better detection.

### Q6: What is the dataset used for music recommendations?
**A:** I created a custom `songs.csv` with 65+ songs manually curated across 7 emotion classes. Each row has: song name, artist, emotion, language (English/Hindi), genre, and YouTube URL. The dataset covers Bollywood, Pop, Rock, Classical, EDM, and more.

### Q7: What are the limitations of your project?
**A:** The main limitations are: (1) Requires good lighting and a front-facing photo for accurate detection, (2) TensorFlow dependency makes installation heavy, (3) Emotion detection has inherent subjectivity — people express emotions differently, (4) The song dataset is small and manually curated.

### Q8: How would you improve this project?
**A:** I would: integrate the Spotify API for real playlist generation, use a larger labeled emotion dataset to fine-tune the model, add a database for user preference learning, and implement real-time video stream processing.

---

## 📃 License

MIT License — free for educational use.

---

*Built with ❤️ as a Final Year Project in Computer Science / AI & ML*
# Music_Recommendation
