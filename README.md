# 🎙️ Whisper Speech-to-Text & Subtitle Generator

An intelligent Speech-to-Text (STT) and subtitle generation application powered by **OpenAI Whisper** and **Streamlit**. It allows you to transcribe audio/video files or live microphone recordings, automatically detect spoken languages, translate speech to English, and export transcriptions in multiple industry-standard formats (TXT, SRT, VTT, JSON).

---

## ✨ Features

- **Multi-Format Input:** Supports audio and video formats including `MP3`, `WAV`, `M4A`, `OGG`, `FLAC`, `MP4`, `MKV`, and `WEBM`.
- **Live Microphone Recording:** Record your voice directly in the browser and transcribe in real-time.
- **Model Scalability:** Switch seamlessly between Whisper model sizes:
  - `tiny`: Lightest and fastest (~39M parameters).
  - `base`: Recommended balance of speed and accuracy (~74M parameters).
  - `small`: Enhanced accuracy for complex vocabulary (~244M parameters).
  - `medium`: High accuracy for multilingual speech (~769M parameters).
- **Language Auto-Detection & Translation:** Transcribe in the original language or translate foreign speech directly into English.
- **Timestamped Subtitle Generation:** Generates synchronized segments with millisecond precision.
- **Multi-Format Export:**
  - 📄 **Plain Text (`.txt`)**
  - 🎬 **SubRip Subtitles (`.srt`)** (for YouTube, VLC, Premiere Pro)
  - 🌐 **WebVTT Subtitles (`.vtt`)** (for HTML5 video players)
  - 📊 **Detailed Metadata (`.json`)**
- **Dual Interface:** Run as an interactive **Web Application** or as an automated **Command-Line Tool (CLI)**.
- **Automatic FFmpeg Management:** Automatically resolves and links FFmpeg on Windows via `imageio-ffmpeg` without manual system PATH edits.

---

## 🏗️ Project Structure

```text
whisper-speech-to-text/
│
├── app.py                # Streamlit Web Application
├── cli.py                # Command-Line Interface tool
├── utils.py              # FFmpeg setup, model loading, SRT/VTT generators
├── create_sample.py      # Quick script to generate test audio
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.10, 3.11, 3.12, 3.13)
- An active Internet connection for initial package and model downloads.

### 2. Setup Virtual Environment (Recommended)

Open PowerShell or Command Prompt in the project folder:

```powershell
cd C:\Users\sinha\.gemini\antigravity\scratch\whisper-speech-to-text

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Windows Command Prompt (CMD):
.\.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

> **Note on PyTorch & GPU:**
> By default, `pip install openai-whisper` installs PyTorch for CPU. If you have an NVIDIA GPU and want blazing-fast GPU acceleration, install CUDA-enabled PyTorch by following [pytorch.org](https://pytorch.org/get-started/locally/):
> ```powershell
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
> ```

---

## 🖥️ Running the Application

### Option A: Launch the Web Application (Streamlit)

With your virtual environment activated, run:

```powershell
streamlit run app.py
```

Your browser will automatically open at `http://localhost:8501`.

1. Choose a Whisper model (e.g. `base`).
2. Select your task: **Transcribe** or **Translate to English**.
3. Upload an audio/video file or use the **Live Microphone** tab.
4. Click **Start Transcription** to view the transcript, inspect timestamped segments, and download `.txt`, `.srt`, `.vtt`, or `.json`.

---

### Option B: Run via Command-Line Interface (CLI)

You can transcribe files directly from your terminal:

```powershell
# Basic transcription (saves .txt, .srt, and .vtt)
python cli.py --file path\to\audio.mp3

# Use a smaller/faster model
python cli.py --file path\to\audio.wav --model tiny

# Translate foreign speech to English and export only SRT subtitles
python cli.py --file interview_es.mp3 --task translate --export srt

# See all available options
python cli.py --help
```

---

## 🧠 How Whisper Works

OpenAI Whisper is an **Encoder-Decoder Transformer**:
1. **Audio Preprocessing:** The input audio is resampled to 16 kHz and converted into an 80-channel log-magnitude Mel spectrogram.
2. **Encoder:** An audio encoder processes the spectrogram chunks into high-dimensional hidden representations.
3. **Decoder:** An autoregressive text decoder predicts text tokens alongside special tokens representing:
   - Voice Activity Detection (speech vs silence)
   - Language Identification
   - Timestamp Prediction (millisecond alignment)
   - Translation directives

| Model Size | Parameters | English-only | Multilingual | VRAM / RAM | Relative Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **tiny** | 39 M | `tiny.en` | `tiny` | ~1 GB | ~32x |
| **base** | 74 M | `base.en` | `base` | ~1 GB | ~16x |
| **small** | 244 M | `small.en` | `small` | ~2 GB | ~6x |
| **medium**| 769 M | `medium.en`| `medium`| ~5 GB | ~2x |
| **large** | 1550 M| N/A | `large` | ~10 GB | 1x |

---

## 🔧 Troubleshooting

- **FFmpeg Not Found:**
  The `utils.py` module includes automatic discovery using `imageio-ffmpeg`. If you ever need manual system FFmpeg, you can install it via Windows Package Manager:
  ```powershell
  winget install Gyan.FFmpeg
  ```
- **Execution Policy Error in PowerShell:**
  If PowerShell blocks activating the virtual environment (`Activate.ps1 cannot be loaded`), run:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

---

## 📜 License & Credits
Built with [OpenAI Whisper](https://github.com/openai/whisper) and [Streamlit](https://streamlit.io/). Open-sourced under the MIT License.
