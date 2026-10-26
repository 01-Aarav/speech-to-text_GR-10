"""
Utility functions for Whisper Speech-to-Text Application.
Handles FFmpeg detection/configuration, model caching, transcription, and subtitle generation.
"""

import os
import sys
import shutil
import datetime
from typing import Dict, Any, List

def setup_ffmpeg() -> str:
    """
    Ensures 'ffmpeg' is accessible in the system PATH.
    If system ffmpeg is missing, locates the binary provided by imageio-ffmpeg
    and ensures a copy/link named 'ffmpeg.exe' exists in that directory.
    """
    existing_ffmpeg = shutil.which("ffmpeg")
    if existing_ffmpeg:
        return existing_ffmpeg

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)

        # Add directory to PATH
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

        # Whisper specifically invokes "ffmpeg" / "ffmpeg.exe"
        target_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        target_path = os.path.join(ffmpeg_dir, target_name)
        if not os.path.exists(target_path) and os.path.exists(ffmpeg_exe):
            try:
                shutil.copy2(ffmpeg_exe, target_path)
            except Exception:
                pass

        return target_path if os.path.exists(target_path) else ffmpeg_exe
    except Exception as exc:
        print(f"[Warning] Failed to configure ffmpeg automatically: {exc}")
        return ""

# Initialize FFmpeg configuration immediately upon import
setup_ffmpeg()

def get_device() -> str:
    """Detects available computing device (CUDA or CPU)."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

def get_models_dir() -> str:
    """Returns local models directory to store Whisper model weights."""
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

def load_model(model_name: str, device: str = None, download_root: str = None):
    """
    Loads OpenAI Whisper model.
    Supported model sizes: 'tiny', 'base', 'small', 'medium', 'large'
    """
    import whisper
    if device is None:
        device = get_device()
    if download_root is None:
        download_root = get_models_dir()
    return whisper.load_model(model_name, device=device, download_root=download_root)

def transcribe_audio(
    model,
    audio_path: str,
    task: str = "transcribe",
    language: str = None,
    temperature: float = 0.0,
    fp16: bool = None
) -> Dict[str, Any]:
    """
    Transcribes or translates the given audio file using Whisper.

    Args:
        model: Loaded Whisper model instance
        audio_path: Path to the audio file
        task: 'transcribe' or 'translate' (translates foreign speech to English)
        language: Optional language code (e.g. 'en', 'es', 'fr', 'hi'). None for auto-detection.
        temperature: Sampling temperature (0.0 is deterministic and best for accuracy)
        fp16: Use FP16 half-precision (True for GPU, False for CPU)
    """
    import torch
    if fp16 is None:
        fp16 = torch.cuda.is_available()

    options = {
        "task": task,
        "temperature": temperature,
        "fp16": fp16,
        "verbose": False
    }

    if language and language != "auto":
        options["language"] = language

    result = model.transcribe(audio_path, **options)
    return result

def format_timestamp(seconds: float, is_vtt: bool = False) -> str:
    """
    Converts seconds into SRT or WebVTT timestamp format.
    SRT: HH:MM:SS,mmm
    VTT: HH:MM:SS.mmm
    """
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)

    delimiter = "." if is_vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{delimiter}{millis:03d}"

def generate_srt(segments: List[Dict[str, Any]]) -> str:
    """Generates standard SubRip Subtitle (.srt) format from Whisper segments."""
    srt_lines = []
    for idx, seg in enumerate(segments, start=1):
        start = format_timestamp(seg["start"], is_vtt=False)
        end = format_timestamp(seg["end"], is_vtt=False)
        text = seg["text"].strip()
        srt_lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
    return "\n".join(srt_lines)

def generate_vtt(segments: List[Dict[str, Any]]) -> str:
    """Generates WebVTT (.vtt) format from Whisper segments."""
    vtt_lines = ["WEBVTT\n"]
    for idx, seg in enumerate(segments, start=1):
        start = format_timestamp(seg["start"], is_vtt=True)
        end = format_timestamp(seg["end"], is_vtt=True)
        text = seg["text"].strip()
        vtt_lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
    return "\n".join(vtt_lines)

# Dictionary of popular languages supported by Whisper
SUPPORTED_LANGUAGES = {
    "Auto-Detect": "auto",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Russian": "ru",
    "Portuguese": "pt",
    "Italian": "it",
    "Dutch": "nl",
    "Turkish": "tr",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Urdu": "ur"
}
