
import re


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if len(sentence.strip().split()) >= 5
    ]


def build_rag_chain(transcript: str):
    """
    Build a lightweight local transcript retriever.
    This fallback requires no Mistral API call or external vector database.
    """
    sentences = _split_sentences(transcript)

    if not sentences:
        raise ValueError("No usable transcript sentences were found.")

    return {"sentences": sentences}


def ask_question(rag_chain, question: str) -> str:
    if not question or not question.strip():
        return "Please enter a question about the lecture."

    if not isinstance(rag_chain, dict):
        raise ValueError("Invalid local lecture search index.")

    sentences = rag_chain.get("sentences", [])
    if not sentences:
        return "There is no transcript available to search."

    stop_words = {
        "the", "and", "for", "that", "this", "with", "what",
        "when", "where", "which", "from", "have", "does",
        "about", "into", "are", "was", "were", "you", "your",
        "can", "could", "would", "tell", "explain", "please",
        "lecture", "main", "topic", "based", "answer",
    }

    question_words = {
        word.lower()
        for word in re.findall(r"\b[a-zA-Z0-9'-]+\b", question)
        if len(word) > 2 and word.lower() not in stop_words
    }

    # Add related terms so broad questions can retrieve relevant content.
    expanded_words = set(question_words)
    if question_words.intersection({"llm", "llms", "model", "models"}):
        expanded_words.update({"large", "language", "model", "models", "llm"})
    if question_words.intersection({"rag", "retrieval", "retriever"}):
        expanded_words.update({
            "rag", "retrieval", "retriever", "augmented",
            "generation", "retrieve", "content", "source",
        })

    ranked = []
    for index, sentence in enumerate(sentences):
        sentence_words = {
            word.lower()
            for word in re.findall(r"\b[a-zA-Z0-9'-]+\b", sentence)
        }

        overlap = len(expanded_words.intersection(sentence_words))
        lower = sentence.lower()

        # Give extra relevance to core concepts when the user asks for the topic.
        if any(term in question.lower() for term in ("main topic", "topic of", "about")):
            if any(term in lower for term in (
                "large language model", "retrieval augmented",
                "retrieval-augmented", "rag", "language models",
            )):
                overlap += 3

        if overlap:
            ranked.append((overlap, index, sentence))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = [item[2] for item in ranked[:5]]

    if not selected:
        return (
            "I couldn't find a reliable answer in the transcript. "
            "Try asking with specific terms mentioned in the lecture."
        )

    return (
        "Based on the lecture transcript:\n\n"
        + "\n".join(f"- {sentence}" for sentence in selected)
    )
