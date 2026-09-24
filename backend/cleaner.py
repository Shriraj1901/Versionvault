import re


def clean_text(text):
    """
    Normalizes messy extracted text before chunking:
    - Collapses excessive whitespace/blank lines
    - Fixes broken mid-sentence line breaks common in PDF extraction
    - Strips weird control characters
    - Normalizes different line-ending styles
    """
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null bytes / control characters (common in bad PDF extracts)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    # Fix hyphenated line-break words e.g. "authenti-\ncation" -> "authentication"
    text = re.sub(r"-\n(?=[a-z])", "", text)

    # Collapse 3+ blank lines into just 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces/tabs into one
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Trim trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return text.strip()