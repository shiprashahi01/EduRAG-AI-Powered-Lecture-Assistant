
import re


def _sentences(transcript: str) -> list[str]:
    text = re.sub(r"\s+", " ", transcript or "").strip()

    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text)
        if len(part.strip().split()) >= 4
    ]


def _clean_items(items: list[str], limit: int = 10) -> list[str]:
    unique = []
    seen = set()

    for item in items:
        item = re.sub(r"\s+", " ", item).strip(" -")
        key = re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()

        if key and key not in seen:
            seen.add(key)
            unique.append(item)

        if len(unique) >= limit:
            break

    return unique


def _bullets(title: str, items: list[str], empty_message: str) -> str:
    if not items:
        return f"### {title}\n\n{empty_message}"

    return f"### {title}\n\n" + "\n".join(
        f"- {item}" for item in items
    )


def extract_action_items(transcript: str) -> str:
    sentences = _sentences(transcript)
    items = []

    # Only recognize explicit tasks or instructions.
    patterns = [
        r"\baction items?\b",
        r"\bassigned to\b",
        r"\bnext steps?\b",
        r"\bdeadline\b",
        r"\bdue date\b",
        r"^(please\s+)?(submit|prepare|complete|finish|create|"
        r"update|review|check|test|install|upload|download|"
        r"implement|fix|build|share|save|record|revise)\b",
        r"\b(i|we|you|they)\s+(will|must|need to|should|"
        r"have to|are going to)\s+(submit|prepare|complete|"
        r"finish|create|update|review|check|test|install|"
        r"upload|download|implement|fix|build|share|save|"
        r"record|revise)\b",
        r"\bmake sure to\s+(submit|prepare|complete|finish|"
        r"create|update|review|check|test|install|upload|"
        r"download|implement|fix|build|share|save|record)\b",
    ]

    for sentence in sentences:
        if any(
            re.search(pattern, sentence, re.IGNORECASE)
            for pattern in patterns
        ):
            items.append(sentence)

    return _bullets(
        "Action Items",
        _clean_items(items, 10),
        "No explicit action items were identified in this lecture.",
    )


def extract_key_decisions(transcript: str) -> str:
    sentences = _sentences(transcript)
    decisions = []

    patterns = [
        r"\bwe decided to\b",
        r"\bthe decision was to\b",
        r"\bit was decided that\b",
        r"\bwe agreed to\b",
        r"\bthe team chose to\b",
        r"\bwe selected\b",
        r"\bthe final decision was\b",
        r"\bwe chose to\b",
        r"\bwe opted to\b",
        r"\bwe settled on\b",
        r"\bwe will proceed with\b",
    ]

    for sentence in sentences:
        if any(
            re.search(pattern, sentence, re.IGNORECASE)
            for pattern in patterns
        ):
            decisions.append(sentence)

    return _bullets(
        "Key Decisions",
        _clean_items(decisions, 10),
        "No explicit decisions or agreements were identified in this lecture.",
    )


def extract_questions(transcript: str) -> str:
    sentences = _sentences(transcript)
    questions = []

    question_start = re.compile(
        r"^(what|why|how|when|where|who|whom|whose|which|"
        r"can|could|would|should|do|does|did|is|are|am|"
        r"was|were|will|has|have|had)\b",
        re.IGNORECASE,
    )

    for sentence in sentences:
        cleaned = sentence.strip(" -")
        words = cleaned.split()

        if len(words) < 5 or len(words) > 45:
            continue

        # A question mark is strongest evidence.
        # Otherwise, require a question-style opening.
        if "?" in cleaned or question_start.search(cleaned):
            questions.append(cleaned)

    return _bullets(
        "Questions",
        _clean_items(questions, 15),
        "No clear questions were detected in the transcript.",
    )
