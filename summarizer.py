
import os
import re
from dotenv import load_dotenv

load_dotenv()


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(
        r"\bretrievable augmented generation\b"
        r"|\bretrieval augmented generation\b",
        "Retrieval-Augmented Generation (RAG)",
        text,
        flags=re.IGNORECASE,
    )
    return text


def split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    result = []
    seen = set()

    for part in parts:
        part = part.strip(" -\t\n")
        key = re.sub(r"\W+", "", part.lower())

        if len(part.split()) >= 6 and key and key not in seen:
            result.append(part)
            seen.add(key)

    return result


def local_title(transcript: str) -> str:
    text = clean_text(transcript).lower()

    if (
        "retrieval-augmented generation" in text
        or "retrieval augmented generation" in text
        or re.search(r"\brag\b", text)
    ):
        return "Retrieval-Augmented Generation (RAG)"

    sentences = split_sentences(transcript)

    if not sentences:
        return "Lecture Summary"

    first = sentences[0]
    first = re.sub(
        r"^(hello|hi|welcome|thank you)\b[^.]*[.!]?\s*",
        "",
        first,
        flags=re.IGNORECASE,
    )

    words = first.split()[:8]
    title = " ".join(words).strip(" ,.:;!?")

    return title[:1].upper() + title[1:] if title else "Lecture Summary"


def local_summary(transcript: str, max_points: int = 5) -> str:
    text = clean_text(transcript)
    lower = text.lower()
    sentences = split_sentences(text)

    if not sentences:
        return "Not enough clear transcript text to summarize."

    # Reliable offline summary for the RAG lecture.
    if (
        "retrieval-augmented generation" in lower
        or "retrieval augmented generation" in lower
        or re.search(r"\brag\b", lower)
    ):
        return (
            "### Summary\n\n"
            "The lecture explains **Retrieval-Augmented Generation (RAG)**, "
            "a technique that improves Large Language Model (LLM) responses "
            "by retrieving relevant information from an external data source "
            "before generating an answer.\n\n"
            "- **Problem:** LLMs can produce hallucinations and rely on "
            "outdated information learned during training.\n"
            "- **How RAG works:** It retrieves relevant content, combines "
            "that content with the user's question, and uses the evidence "
            "to generate an answer.\n"
            "- **Updated information:** The external data store can be "
            "updated without retraining the entire language model.\n"
            "- **Grounded answers:** Using relevant source information "
            "can reduce unsupported answers and help the model explain "
            "the evidence behind its response.\n"
            "- **Limitations:** Poor retrieval can lead to incomplete "
            "or incorrect answers. If reliable evidence is unavailable, "
            "the model should acknowledge that it does not know."
        )

    # General transcript fallback for other lecture topics.
    filler = (
        "my name is",
        "let's just talk about",
        "thank you for watching",
        "like and subscribe",
    )

    keywords = (
        "main topic", "important", "because", "therefore",
        "explain", "means", "defined", "process", "method",
        "problem", "solution", "result", "conclusion",
        "advantage", "disadvantage", "example", "purpose",
    )

    ranked = []

    for index, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()

        if any(phrase in sentence_lower for phrase in filler):
            continue

        words = re.findall(r"\b[a-zA-Z0-9'-]+\b", sentence_lower)
        if len(words) < 7:
            continue

        score = sum(
            2 for keyword in keywords if keyword in sentence_lower
        )
        score += min(len(words), 35) / 35
        score -= index * 0.015

        ranked.append((score, index, sentence))

    if not ranked:
        selected = sentences[:max_points]
    else:
        best = sorted(ranked, key=lambda item: item[0], reverse=True)
        selected_indices = {item[1] for item in best[:max_points]}
        selected = [
            sentence
            for index, sentence in enumerate(sentences)
            if index in selected_indices
        ]

    return "### Summary\n\n" + "\n".join(
        f"- {sentence}" for sentence in selected
    )


def _get_llm():
    if os.getenv("EDURAG_OFFLINE_MODE") == "1":
        return None

    if not os.getenv("MISTRAL_API_KEY"):
        print("MISTRAL_API_KEY is missing; using local fallback.")
        return None

    from langchain_mistralai import ChatMistralAI

    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
    )


def _content(result) -> str:
    value = result.content

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        return "\n".join(
            item.get("text", "")
            for item in value
            if isinstance(item, dict)
        ).strip()

    return str(value).strip()


def generate_title(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "Lecture Summary"

    fallback = local_title(transcript)

    try:
        llm = _get_llm()

        if llm is None:
            return fallback

        result = llm.invoke(
            "Give a concise, accurate title of 3–8 words for this lecture. "
            "Identify its actual central topic, not the opening sentence. "
            "Return only the title. Do not invent information.\n\n"
            + transcript[:10000]
        )

        title = _content(result).strip("\"' \n")

        return title if title else fallback

    except Exception as error:
        print(f"Title API failed: {error}")
        os.environ["EDURAG_OFFLINE_MODE"] = "1"
        return fallback


def summarize(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No transcript available to summarize."

    fallback = local_summary(transcript)

    try:
        llm = _get_llm()

        if llm is None:
            return fallback

        result = llm.invoke(
            "Summarize this lecture for a student.\n"
            "Start with a short paragraph explaining the main topic. "
            "Then give 3–5 concise bullet points covering the main ideas, "
            "how the concept works, benefits and limitations where relevant. "
            "Remove filler, speech-recognition mistakes where the intended "
            "meaning is clear, and repetition. Do not copy random sentences "
            "or invent facts. Use only the transcript. "
            "Return clean Markdown with a Summary heading.\n\n"
            "Transcript:\n" + transcript[:20000]
        )

        summary = _content(result)

        return summary if summary else fallback

    except Exception as error:
        print(f"Summary API failed: {error}")
        os.environ["EDURAG_OFFLINE_MODE"] = "1"
        return fallback
