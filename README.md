# 🎬 Hacklanta

**AI-powered media analysis framework** with dual-processing architecture — local ML models first, with LLM API fallback for higher-order reasoning.

Built for the [HackATL / Hacklanta](https://hackatl.org) hackathon.

---

## ✨ What It Does

Hacklanta provides specialized **AI agents** that analyze audio and visual media:

| Agent | Local Processing | API Fallback |
|---|---|---|
| **AudioMatchAgent** | BPM detection, energy analysis, spectral features (Librosa) | Genre/mood classification via GPT |
| **StyleDirectorAgent** | Color histogram matching, K-means palette extraction, contrast/saturation/temperature analysis (OpenCV) | Refined aesthetic insights via GPT |

Each agent follows a **confidence-gated dual-processing** pattern:
1. Run local ML analysis first (fast, free, offline)
2. Score confidence of the local result
3. If confidence is below threshold → fallback to OpenRouter LLM API

---

## 📁 Project Structure

```
Hacklanta/
├── agents/                     # AI Agent framework
│   ├── __init__.py
│   ├── base_agent.py           # Abstract base with dual-processing logic
│   ├── audio_match_agent.py    # Audio analysis (BPM, energy, genre/mood)
│   └── style_director.py       # Image style analysis & LUT recommendations
├── openrouter_client/          # OpenRouter API client library
│   ├── __init__.py
│   ├── client.py               # Main client (chat completions, retries)
│   ├── cache.py                # Response caching with similarity hashing
│   ├── cost_tracker.py         # API cost tracking & analytics
│   ├── rate_limiter.py         # Token bucket rate limiter
│   └── exceptions.py           # Custom exception hierarchy
├── test_agents.ipynb           # Interactive testing notebook
├── ml_setup_verification.py    # Verify local ML dependencies
├── requirements.txt            # Python dependencies
├── secrets.env                 # API keys (gitignored)
├── yolov8n.pt                  # YOLOv8-nano weights
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/Tushar-Tyagi/Hacklanta.git
cd Hacklanta
pip install -r requirements.txt
```

### 2. Add Your API Key

Edit `secrets.env` and add your [OpenRouter](https://openrouter.ai) API key:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

> **Note:** `secrets.env` is gitignored and will never be committed.

### 3. Verify ML Setup

```bash
python ml_setup_verification.py
```

This checks that YOLOv8, DeepFace, MediaPipe, Librosa, and OpenCV are installed correctly.

### 4. Run the Test Notebook

```bash
jupyter notebook test_agents.ipynb
```

---

<<<<<<< Updated upstream
## 🖥️ How to test the frontend UI

1. Start the backend Flask server:
```bash
python app.py
```
2. Open your web browser and navigate to `http://localhost:5000` (or `http://127.0.0.1:5000`).
3. You will see the Video Style Director interface.
4. Upload files:
   - Click "Choose File" under "Upload Video (Required)" to select a video file (e.g., .mp4, .mkv, .avi, .mov).
   - Click "Choose File" under "Upload Audio (Optional)" if you want to test audio synchronization features (e.g., .mp3, .wav).
5. Click the "Process Media" button.
6. The UI will show a loading spinner while the backend mock agents process the files.
7. Once processing is complete, you will see the results populated in three different tabs:
   - **Scenes**: Shows the detected scene cuts and types.
   - **Style**: Displays the recommended cinematic style, color grading notes, and color palette.
   - **Audio & Composition**: Shows audio analysis (BPM, mood) and edit decisions (if audio was provided).

> Note: The current backend relies on mock processing for demonstration purposes, so processing will be quick and return pre-computed example data.

---

=======
>>>>>>> Stashed changes
## 🔧 Usage

### Processing Modes

All agents support four processing modes:

```python
from agents import ProcessingMode

ProcessingMode.LOCAL_ONLY     # Local ML models only (no API calls)
ProcessingMode.API_ONLY       # OpenRouter API only
ProcessingMode.HYBRID         # Local first → API if confidence < threshold
ProcessingMode.API_FALLBACK   # Local first → API on failure
```

### AudioMatchAgent

Analyzes audio files for BPM, energy, spectral features, genre, and mood.

```python
from agents import AudioMatchAgent, ProcessingMode

agent = AudioMatchAgent(
    api_key="your-openrouter-key",
    mode=ProcessingMode.HYBRID,
)

# Extract low-level audio features (always local)
features = agent.analyze_audio_file("song.mp3")
print(f"BPM: {features.bpm}, Energy: {features.energy_level}")
print(f"Timbre: {features.timbre}, Tempo: {features.tempo_category}")

# Classify genre & mood (local + optional API fallback)
result = agent.classify_genre_mood("song.mp3")
print(f"Genres: {result.genres}")      # e.g. ['pop', 'rock']
print(f"Moods: {result.moods}")        # e.g. ['energetic', 'intense']
print(f"Confidence: {result.confidence:.2f}")
print(f"Source: {result.source}")      # 'local' or 'api'
```

**Supported formats:** `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.wma`

### StyleDirectorAgent

Analyzes image aesthetics — colors, mood, contrast, temperature — and recommends cinematic LUTs.

```python
from agents import StyleDirectorAgent, ProcessingMode

agent = StyleDirectorAgent(
    api_key="your-openrouter-key",
    mode=ProcessingMode.LOCAL_ONLY,
)

# Analyze image style
result = agent.analyze_style("photo.jpg")
print(f"Mood: {result.mood}")                 # e.g. 'warm'
print(f"Temperature: {result.temperature}")    # 'cool', 'neutral', 'warm'
print(f"Contrast: {result.contrast_level}")    # 'low', 'medium', 'high'
print(f"Saturation: {result.saturation_level}")# 'muted', 'moderate', 'vibrant'
print(f"Primary colors: {result.primary_colors}")

# Compare two images
result = agent.analyze_style("image1.jpg", reference_path="image2.jpg")
print(f"Histogram similarity: {result.histogram_similarity:.3f}")

# Style transfer via histogram matching
output = agent.apply_style_transfer("source.jpg", "reference.jpg")
# Saves to source_styled.jpg

# Get cinematic LUT recommendations
luts = agent.get_lut_recommendations(result)
for lut in luts:
    print(f"{lut.name} ({lut.category}) — {lut.description}")
```

### OpenRouter Client (standalone)

The `openrouter_client` package can be used independently:

```python
from openrouter_client import OpenRouterClient

client = OpenRouterClient(
    api_key="your-key",
    enable_caching=True,
    enable_cost_tracking=True,
)

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    temperature=0.7,
    max_tokens=256,
)

print(response.get_content())
client.close()
```

**Features:**
- Rate limiting (token bucket algorithm)
- Automatic retry with exponential backoff
- Response caching with similarity-based hashing
- Cost tracking and analytics
- Comprehensive error handling (auth, rate limits, credits, timeouts)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│                  BaseAgent                   │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Process │──▶│Confidence│──▶│ Fallback │ │
│  │ (Local) │   │  Score   │   │  (API)   │ │
│  └─────────┘   └──────────┘   └──────────┘ │
│       │              │              │        │
│  ┌────▼────┐   ┌─────▼─────┐  ┌────▼─────┐ │
│  │  Cache  │   │ Threshold │  │OpenRouter│ │
│  │  Layer  │   │   Check   │  │  Client  │ │
│  └─────────┘   └───────────┘  └──────────┘ │
└─────────────────────────────────────────────┘
         │                            │
    ┌────▼─────┐              ┌───────▼───────┐
    │  Librosa │              │  Rate Limiter  │
    │  OpenCV  │              │  Cost Tracker  │
    │  YOLO    │              │  Retry Logic   │
    │MediaPipe │              │  Resp. Cache   │
    └──────────┘              └───────────────┘
     Local ML                   OpenRouter API
```

---

## 📋 Dependencies

| Category | Packages |
|---|---|
| **Object Detection** | `ultralytics` (YOLOv8), `torch`, `torchvision` |
| **Face Recognition** | `deepface` |
| **Body/Hand Tracking** | `mediapipe` |
| **Audio Analysis** | `librosa` |
| **Image Processing** | `opencv-python`, `Pillow` |
| **Numerical** | `numpy`, `pandas`, `scipy` |
| **Video Editing** | `moviepy` |
| **HTTP/API** | `requests` |

See [`requirements.txt`](requirements.txt) for full version constraints and platform-specific notes.

---

## 📄 License

This project was built for the Hacklanta hackathon.
