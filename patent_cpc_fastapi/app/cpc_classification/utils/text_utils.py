"""text_utils.py — Text normalisation and truncation utilities."""


def normalize_word(word: str) -> str:
    """Normalize a word for matching: lowercase, strip punctuation, basic stemming."""
    word = word.lower().strip(".,;:!?()[]{}-")
    if word.endswith("ing") and len(word) > 5:
        word = word[:-3]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("s") and len(word) > 3:
        word = word[:-1]
    elif word.endswith("es") and len(word) > 4:
        word = word[:-2]
    return word


def shorten(text: str, max_len: int = 80) -> str:
    """Truncate text to max_len chars, adding ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
