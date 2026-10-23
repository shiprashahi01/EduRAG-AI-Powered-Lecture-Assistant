
import os
import re
import uuid
from pathlib import Path

import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    if not re.match(r"^https?://(www\.)?(youtube\.com|youtu\.be)/", url.strip()):
        raise ValueError("Please enter a valid YouTube video URL.")

    output_path = os.path.abspath(
        os.path.join(DOWNLOAD_DIR, f"%(id)s_{uuid.uuid4().hex[:8]}.%(ext)s")
    )

    options = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "noplaylist": True,
        "retries": 3,
        "fragment_retries": 3,
        "quiet": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url.strip(), download=True)
        filename = ydl.prepare_filename(info)
        wav_path = os.path.splitext(filename)[0] + ".wav"

    if not os.path.isfile(wav_path):
        raise FileNotFoundError(
            "YouTube audio conversion failed. Check that FFmpeg is installed."
        )
    return wav_path


def convert_to_wav(input_path: str) -> str:
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 5) -> list[str]:
    if chunk_minutes <= 0:
        raise ValueError("chunk_minutes must be greater than zero.")

    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk_path = (
            f"{os.path.splitext(wav_path)[0]}_chunk_{i}.wav"
        )
        audio[start:start + chunk_ms].export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source) -> list[str]:
    if isinstance(source, str) and source.strip().startswith(("http://", "https://")):
        print("Downloading YouTube audio...")
        wav_path = download_youtube_audio(source.strip())
    else:
        # Supports both a Streamlit UploadedFile and a local file path.
        if hasattr(source, "getvalue") and hasattr(source, "name"):
            suffix = Path(source.name).suffix or ".mp4"
            input_path = os.path.abspath(
                os.path.join(DOWNLOAD_DIR, f"upload_{uuid.uuid4().hex}{suffix}")
            )
            with open(input_path, "wb") as file:
                file.write(source.getvalue())
        elif isinstance(source, str) and os.path.isfile(source):
            input_path = source
        else:
            raise ValueError("Please upload a valid audio/video file or enter a YouTube URL.")

        print("Converting uploaded audio/video...")
        wav_path = convert_to_wav(input_path)

    print("Splitting audio into chunks...")
    chunks = chunk_audio(wav_path)

    if not chunks:
        raise ValueError("No audio chunks were created.")

    print(f"Audio ready: {len(chunks)} chunk(s).")
    return chunks
