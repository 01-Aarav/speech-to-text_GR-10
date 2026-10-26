"""
Whisper Speech-to-Text & Translation Web Application
Built with OpenAI Whisper and Streamlit
"""

import os
import tempfile
import time
import json
import hashlib
import streamlit as st

# Import utilities
from utils import (
    load_model,
    transcribe_audio,
    generate_srt,
    generate_vtt,
    get_device,
    setup_ffmpeg,
    SUPPORTED_LANGUAGES,
)

# Page configuration
st.set_page_config(
    page_title="Whisper Speech-to-Text Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stDownloadButton > button {
        width: 100%;
        border-radius: 8px;
    }
    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Ensure FFmpeg is configured
ffmpeg_path = setup_ffmpeg()

# Initialize session state keys for dynamic widget resetting
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "mic_key" not in st.session_state:
    st.session_state["mic_key"] = 0
if "last_audio_hash" not in st.session_state:
    st.session_state["last_audio_hash"] = None

def reset_all_audio():
    """Resets all audio inputs, cached state, and transcription results."""
    st.session_state["uploader_key"] += 1
    st.session_state["mic_key"] += 1
    st.session_state.pop("whisper_result", None)
    st.session_state.pop("elapsed_time", None)
    st.session_state.pop("processed_filename", None)
    st.session_state.pop("last_audio_hash", None)

# Model Specifications
MODEL_INFO = {
    "tiny": {"params": "39M", "vram": "~1 GB", "speed": "Very Fast (~32x)", "desc": "Lightest model, cached locally and great for quick tests."},
    "base": {"params": "74M", "vram": "~1 GB", "speed": "Fast (~16x)", "desc": "Default sweet spot between speed and transcription quality."},
    "small": {"params": "244M", "vram": "~2 GB", "speed": "Moderate (~6x)", "desc": "High accuracy for complex vocabulary and noisy audio."},
    "medium": {"params": "769M", "vram": "~5 GB", "speed": "Slow (~2x)", "desc": "High accuracy, best suited for multilingual audio and GPU."},
}

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Configuration")

    device = get_device()
    st.info(f"**Hardware Acceleration:** `{device.upper()}`")

    model_name = st.selectbox(
        "Whisper Model Size",
        options=["tiny", "base", "small", "medium"],
        index=1,
        help="Larger models provide higher accuracy but require more memory and time."
    )

    info = MODEL_INFO[model_name]
    st.caption(f"**Parameters:** {info['params']} | **VRAM/RAM:** {info['vram']}\n\n**Speed:** {info['speed']}\n\n*{info['desc']}*")

    st.markdown("---")
    st.subheader("🌐 Task & Language")

    task = st.radio(
        "Task Mode",
        options=["transcribe", "translate"],
        format_func=lambda x: "📝 Transcribe (Speech to Text)" if x == "transcribe" else "🌐 Translate (Translate into English)",
        help="'Transcribe' keeps the spoken language. 'Translate' automatically translates any foreign language into English."
    )

    selected_language_name = st.selectbox(
        "Spoken Language",
        options=list(SUPPORTED_LANGUAGES.keys()),
        index=0,
        help="Leave as 'Auto-Detect' unless you want to force a specific language."
    )
    language_code = SUPPORTED_LANGUAGES[selected_language_name]

    st.markdown("---")
    st.subheader("🧹 Actions")
    if st.button("🗑️ Reset Audio & Results", use_container_width=True, type="secondary"):
        reset_all_audio()
        st.rerun()

    st.caption("Powered by **OpenAI Whisper** & **Streamlit**")


@st.cache_resource(show_spinner=False)
def get_cached_whisper_model(name: str, dev: str):
    """Loads and caches the Whisper model in memory so it doesn't reload each time."""
    return load_model(name, device=dev)


# Main Content Header
st.markdown('<div class="main-header">🎙️ Whisper Speech-to-Text Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Convert voice recordings, audio clips, and videos into accurate text and subtitle files using OpenAI Whisper.</div>',
    unsafe_allow_html=True,
)

# Input Mode Selector (Radio keeps the two methods isolated so live mic doesn't overwrite file uploads)
input_mode = st.radio(
    "Choose Input Method:",
    options=["📁 Upload Audio / Video File", "🎤 Live Microphone Recording"],
    horizontal=True,
)

audio_bytes = None
audio_file_name = "audio.wav"

if input_mode == "📁 Upload Audio / Video File":
    uploaded_file = st.file_uploader(
        "Upload audio or video file",
        type=["mp3", "wav", "m4a", "ogg", "flac", "mp4", "webm", "mkv"],
        help="Supported formats: MP3, WAV, M4A, OGG, FLAC, MP4, WEBM, MKV",
        key=f"uploader_{st.session_state.uploader_key}"
    )
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        audio_file_name = uploaded_file.name

elif input_mode == "🎤 Live Microphone Recording":
    st.write("Record your voice directly from your microphone:")
    if hasattr(st, "audio_input"):
        mic_audio = st.audio_input("Record audio", key=f"mic_{st.session_state.mic_key}")
        if mic_audio is not None:
            audio_bytes = mic_audio.read()
            audio_file_name = "microphone_recording.wav"
    else:
        st.warning("Live audio input requires Streamlit 1.40+. Please use the file upload option.")

# Automatic detection of audio change: clear old results if audio changed
if audio_bytes is not None:
    current_audio_hash = hashlib.md5(audio_bytes).hexdigest()
    if st.session_state["last_audio_hash"] != current_audio_hash:
        st.session_state["last_audio_hash"] = current_audio_hash
        st.session_state.pop("whisper_result", None)
        st.session_state.pop("elapsed_time", None)

# Process Audio Section
if audio_bytes is not None:
    st.write("---")
    col_audio, col_action = st.columns([1.6, 1.4])

    with col_audio:
        st.subheader("🎧 Audio Preview")
        st.caption(f"**Loaded File:** `{audio_file_name}` ({len(audio_bytes) / 1024:.1f} KB)")
        st.audio(audio_bytes)

    run_triggered = False

    with col_action:
        st.subheader("🚀 Transcription Controls")
        has_result = "whisper_result" in st.session_state

        if has_result:
            st.success("✅ Transcription Complete!")
            st.caption("You can inspect the results below, re-run with different settings, or reset.")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                run_triggered = st.button("🔁 Re-transcribe", type="secondary", use_container_width=True)
            with col_b2:
                if st.button("🗑️ Reset Audio", type="primary", use_container_width=True):
                    reset_all_audio()
                    st.rerun()
        else:
            action_label = "Start Transcription" if task == "transcribe" else "Start Translation"
            col_b1, col_b2 = st.columns([2, 1])
            with col_b1:
                run_triggered = st.button(f"▶️ {action_label}", type="primary", use_container_width=True)
            with col_b2:
                if st.button("🗑️ Reset", type="secondary", use_container_width=True):
                    reset_all_audio()
                    st.rerun()

    if run_triggered:
        file_ext = os.path.splitext(audio_file_name)[1]
        if not file_ext:
            file_ext = ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = temp_audio.name

        try:
            with st.spinner(f"Loading '{model_name}' model and processing speech..."):
                start_time = time.time()

                model = get_cached_whisper_model(model_name, device)
                lang_arg = None if language_code == "auto" else language_code

                result = transcribe_audio(
                    model=model,
                    audio_path=temp_path,
                    task=task,
                    language=lang_arg
                )

                elapsed_time = time.time() - start_time

            # Store results in session state
            st.session_state["whisper_result"] = result
            st.session_state["elapsed_time"] = elapsed_time
            st.session_state["processed_filename"] = audio_file_name
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred during transcription: {e}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

# Display Results
if "whisper_result" in st.session_state:
    result = st.session_state["whisper_result"]
    elapsed_time = st.session_state["elapsed_time"]
    filename = st.session_state.get("processed_filename", "audio")
    base_name = os.path.splitext(filename)[0]

    st.write("---")
    res_header_col1, res_header_col2 = st.columns([3, 1])
    with res_header_col1:
        st.subheader("📊 Results & Analytics")
    with res_header_col2:
        if st.button("➕ New Audio / Reset", use_container_width=True):
            reset_all_audio()
            st.rerun()

    detected_lang = result.get("language", "Unknown")
    full_text = result.get("text", "").strip()
    segments = result.get("segments", [])
    total_words = len(full_text.split())

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{detected_lang.upper()}</div><div class="metric-label">Detected Language</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{elapsed_time:.2f}s</div><div class="metric-label">Processing Time</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{total_words}</div><div class="metric-label">Total Words</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{len(segments)}</div><div class="metric-label">Segments</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")

    # Transcript Tabs
    t_text, t_segments, t_raw = st.tabs(["📝 Full Text", "⏱️ Timestamped Segments", "⚙️ Raw JSON"])

    with t_text:
        st.text_area("Transcription Output", value=full_text, height=260)

    with t_segments:
        if segments:
            st.caption("Detected speech chunks with millisecond timestamps:")
            for seg in segments:
                start_m, start_s = divmod(seg["start"], 60)
                end_m, end_s = divmod(seg["end"], 60)
                timestamp_badge = f"{int(start_m):02d}:{start_s:05.2f} ➔ {int(end_m):02d}:{end_s:05.2f}"
                with st.container():
                    st.markdown(f"**`{timestamp_badge}`** &nbsp; {seg['text'].strip()}")
        else:
            st.info("No distinct speech segments detected in this audio.")

    with t_raw:
        st.json(result)

    # Export & Download Section
    st.subheader("📥 Export & Download")
    srt_content = generate_srt(segments)
    vtt_content = generate_vtt(segments)
    json_content = json.dumps(result, indent=2, ensure_ascii=False)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button(
            label="📄 Plain Text (.txt)",
            data=full_text,
            file_name=f"{base_name}_transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            label="🎬 SubRip Subtitle (.srt)",
            data=srt_content,
            file_name=f"{base_name}.srt",
            mime="text/plain",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            label="🌐 WebVTT Subtitle (.vtt)",
            data=vtt_content,
            file_name=f"{base_name}.vtt",
            mime="text/vtt",
            use_container_width=True,
        )
    with d4:
        st.download_button(
            label="📊 Full JSON (.json)",
            data=json_content,
            file_name=f"{base_name}_whisper.json",
            mime="application/json",
            use_container_width=True,
        )
else:
    if audio_bytes is None:
        st.info("👆 Upload an audio file or record using your microphone above to begin.")
