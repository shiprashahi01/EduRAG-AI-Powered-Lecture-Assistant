import os
import requests
import whisper
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")
SARVAM_PIECE_SECONDS = 25

_model = None


def load_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}")
        _model = whisper.load_model(WHISPER_MODEL)
    return _model


def transcribe_chunk_whisper(chunk_path: str, language: str = "auto") -> str:
    model = load_model()
    options = {"task": "transcribe"}
    if language.lower() in ("english", "hindi"):
        options["language"] = language.lower()

    result = model.transcribe(chunk_path, **options)
    return result.get("text", "").strip()


def _send_to_sarvam(piece_path: str) -> str:
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is missing from your .env file.")

    headers = {"api-subscription-key": SARVAM_API_KEY}
    with open(piece_path, "rb") as audio_file:
        response = requests.post(
            SARVAM_URL,
            headers=headers,
            files={"file": (os.path.basename(piece_path), audio_file, "audio/wav")},
            data={"model": SARVAM_MODEL},
            timeout=120,
        )

    if not response.ok:
        raise RuntimeError(
            f"Sarvam API error {response.status_code}: {response.text[:400]}"
        )

    result = response.json()
    transcript = result.get("transcript", "")
    if not transcript:
        raise RuntimeError("Sarvam returned an empty transcript.")
    return transcript.strip()


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is missing from your .env file.")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    results = []

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece_path = f"{chunk_path}_sarvam_{i}.wav"
        try:
            audio[start:start + piece_ms].export(piece_path, format="wav")
            print(f"Sarvam piece {i + 1}...")
            results.append(_send_to_sarvam(piece_path))
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return " ".join(results).strip()


def transcribe_chunk(chunk_path: str, language: str = "auto") -> str:
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path, language)


def transcribe_all(chunks: list[str], language: str = "auto") -> str:
    if not chunks:
        raise ValueError("No audio chunks were provided for transcription.")

    results = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"Transcribing chunk {i}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language)
        if text:
            results.append(text)

    transcript = " ".join(results).strip()
    if not transcript:
        raise RuntimeError("Transcription returned no text.")
    return transcript
