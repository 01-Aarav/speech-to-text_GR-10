"""
Command-Line Interface (CLI) for OpenAI Whisper Speech-to-Text.
Usage:
    python cli.py --file path/to/audio.mp3 --model base
    python cli.py --file interview.wav --task translate --export srt
"""

import argparse
import os
import sys
import time

# Ensure Windows console supports UTF-8 characters and emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from utils import (
    setup_ffmpeg,
    get_device,
    load_model,
    transcribe_audio,
    generate_srt,
    generate_vtt,
)

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe or translate speech to text using OpenAI Whisper"
    )
    parser.add_argument(
        "-f", "--file",
        required=True,
        help="Path to the input audio or video file"
    )
    parser.add_argument(
        "-m", "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "-t", "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="'transcribe' preserves language; 'translate' translates to English"
    )
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Spoken language code (e.g. en, fr, es, hi). Omit for auto-detect."
    )
    parser.add_argument(
        "-e", "--export",
        choices=["txt", "srt", "vtt", "all"],
        default="all",
        help="Output format to export (default: all)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory to save generated output files (default: same as input file)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: Input file '{args.file}' not found.", file=sys.stderr)
        sys.exit(1)

    setup_ffmpeg()
    device = get_device()

    print("=" * 60)
    print("🎙️ OpenAI Whisper Speech-to-Text CLI")
    print(f"  • Input File : {args.file}")
    print(f"  • Model      : {args.model}")
    print(f"  • Device     : {device.upper()}")
    print(f"  • Task       : {args.task}")
    print("=" * 60)

    print(f"Loading '{args.model}' model...")
    model = load_model(args.model, device=device)

    print("Transcribing audio...")
    start_time = time.time()
    result = transcribe_audio(
        model=model,
        audio_path=args.file,
        task=args.task,
        language=args.language
    )
    elapsed = time.time() - start_time

    detected_lang = result.get("language", "unknown")
    full_text = result.get("text", "").strip()
    segments = result.get("segments", [])

    print("\n" + "=" * 60)
    print(f"✅ Completed in {elapsed:.2f} seconds")
    print(f"🌐 Detected Language: {detected_lang}")
    print(f"📊 Segments: {len(segments)}")
    print("=" * 60)
    print("\n[TRANSCRIPTION]:\n")
    print(full_text)
    print("\n" + "-" * 60)

    # Export results
    output_dir = args.output_dir if args.output_dir else os.path.dirname(os.path.abspath(args.file))
    base_name = os.path.splitext(os.path.basename(args.file))[0]

    if args.export in ["txt", "all"]:
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text + "\n")
        print(f"Saved: {txt_path}")

    if args.export in ["srt", "all"]:
        srt_path = os.path.join(output_dir, f"{base_name}.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(generate_srt(segments))
        print(f"Saved: {srt_path}")

    if args.export in ["vtt", "all"]:
        vtt_path = os.path.join(output_dir, f"{base_name}.vtt")
        with open(vtt_path, "w", encoding="utf-8") as f:
            f.write(generate_vtt(segments))
        print(f"Saved: {vtt_path}")

if __name__ == "__main__":
    main()
