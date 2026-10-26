"""
Generates a sample spoken English WAV file for testing Whisper transcription.
Uses Windows built-in speech synthesizer or a fallback sound generator.
"""

import os
import subprocess
import sys

def generate_sample_wav(output_path: str = "sample_speech.wav") -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(output_path):
        output_path = os.path.join(script_dir, output_path)
    abs_output = os.path.abspath(output_path)
    text_to_speak = (
        "Hello and welcome! This is a test recording for the speech to text converter project. "
        "OpenAI Whisper uses deep learning transformer models to accurately transcribe spoken words into text, "
        "detect languages, and generate subtitle files with precise timestamps."
    )

    if sys.platform == "win32":
        ps_command = f"""
        try {{
            Add-Type -AssemblyName System.Speech
            $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
            $synth.SetOutputToWaveFile('{abs_output}')
            $synth.Speak('{text_to_speak}')
            $synth.Dispose()
        }} catch {{
            exit 1
        }}
        """
        try:
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            if res.returncode == 0 and os.path.exists(abs_output) and os.path.getsize(abs_output) > 1000:
                print(f"Generated sample spoken audio at: {abs_output}")
                return abs_output
        except Exception:
            pass

    # Fallback to pure python sine-wave audio if speech synthesis fails
    import wave
    import math
    import struct

    sample_rate = 16000
    duration_secs = 3
    num_samples = sample_rate * duration_secs

    with wave.open(abs_output, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for i in range(num_samples):
            value = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
            data = struct.pack("<h", value)
            wav_file.writeframesraw(data)

    print(f"Generated synthetic audio fallback at: {abs_output}")
    return abs_output

if __name__ == "__main__":
    generate_sample_wav()
