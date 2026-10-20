# EduRAG-AI-Powered-Lecture-Assistant
AI-powered lecture assistant that transcribes lectures, generates summaries and key learning points, and answers questions using Retrieval-Augmented Generation (RAG).

## Features

- **Lecture Input:** Process YouTube videos and uploaded audio/video files.
- **Speech-to-Text:** Convert lecture audio into text using Whisper.
- **Translation:** Support Hindi-to-English transcription/translation through the configured speech service.
- **Lecture Summarization:** Generate concise summaries of lecture content.
- **Key Learning Points:** Extract important concepts and study points.
- **AI Question Answering:** Ask questions about lecture content using Retrieval-Augmented Generation (RAG).
- **Semantic Search:** Store and retrieve relevant lecture content using embeddings and ChromaDB.
- **Export:** Save study material as PDF or TXT.

## Tech Stack

- Python
- Streamlit
- Whisper
- LangChain
- Mistral API
- Hugging Face Sentence Transformers
- ChromaDB
- yt-dlp
- FFmpeg / pydub

## How It Works

1. Provide a lecture video URL or upload a supported media file.
2. Extract and process the audio.
3. Transcribe the lecture into text.
4. Generate a summary and key learning points.
5. Store transcript chunks as embeddings in ChromaDB.
6. Ask questions and receive answers based on the lecture content.
7. Export the generated study material.

## Installation

Clone the repository and open its folder in a terminal.

```bash
python -m venv .venv
```

Activate the virtual environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
pip install -r Requirements.txt
```

Create a `.env` file and add the API key required by your configured services. Never commit API keys to GitHub.

Run the Streamlit application:

```powershell
streamlit run app.py
```

## Project Objective

The objective of EduRAG is to make lecture revision easier by transforming lengthy educational content into searchable, concise, and interactive study material.

## Project Status

Educational AI/ML project developed in the context of an internship and college project.

*Note: Available features depend on the configured API services and the current project implementation.*
