"""Label cleaning utilities for visual generation."""

import re
from typing import Set

# Generic stopwords to strip from labels
STOPWORDS: Set[str] = {
    "total", "amount", "approximate", "mentioned", "targeted", "aiming",
    "earlier", "report", "of", "the", "by", "to", "in", "a", "an",
    "and", "or", "for", "with", "on", "at", "from", "as", "is", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "must",
    "this", "that", "these", "those", "it", "its", "their", "his", "her",
    "our", "your", "my", "i", "we", "you", "he", "she", "they", "them",
    "us", "me", "him", "her", "what", "which", "who", "whom", "whose",
    "where", "when", "why", "how", "all", "each", "every", "some", "any",
    "no", "not", "only", "just", "also", "even", "still", "yet", "already",
    "ever", "never", "always", "often", "sometimes", "usually", "generally",
    "specifically", "particularly", "especially", "mainly", "mostly",
    "largely", "partly", "partially", "fully", "completely", "entirely",
    "wholly", "totally", "absolutely", "definitely", "certainly", "surely",
    "probably", "possibly", "likely", "unlikely", "maybe", "perhaps",
    "seemingly", "apparently", "evidently", "obviously", "clearly",
    "plainly", "simply", "merely", "barely", "hardly", "scarcely",
    "about", "above", "below", "between", "beyond", "under", "over",
    "through", "during", "before", "after", "since", "until", "while",
    "when", "where", "while", "whereas", "whereby", "whereupon",
    "therefore", "thus", "hence", "consequently", "accordingly",
    "furthermore", "moreover", "additionally", "besides", "similarly",
    "likewise", "conversely", "however", "nevertheless", "nonetheless",
    "otherwise", "instead", "rather", "otherwise", "otherwise", "else",
    "anyway", "anyhow", "anywhere", "everywhere", "nowhere", "somewhere",
    "someone", "somebody", "something", "anyone", "anybody", "anything",
    "everyone", "everybody", "everything", "noone", "nobody", "nothing",
    "each", "other", "another", "such", "same", "different", "various",
    "several", "many", "few", "little", "much", "more", "most", "less",
    "least", "enough", "too", "very", "quite", "rather", "pretty", "fairly",
    "somewhat", "slightly", "a bit", "a lot", "lots", "plenty", "enough",
    "much", "many", "few", "little", "less", "least", "more", "most",
}

# Units and currency symbols to preserve
UNIT_PATTERNS = [
    r"\$", r"€", r"£", r"¥", r"%", r"B\b", r"M\b", r"K\b", r"T\b",
    r"ms\b", r"GB\b", r"TB\b", r"MB\b", r"KB\b", r"tokens?\b",
    r"users?\b", r"params?\b", r"FLOPs?\b", r"TFLOPs?\b",
]


def clean_label(raw: str, max_len: int = 18) -> str:
    """
    Clean and truncate a label for chart axes.

    Args:
        raw: Raw label text from data
        max_len: Maximum length (default 18 chars)

    Returns:
        Cleaned, Title Case label truncated on word boundary
    """
    if not raw:
        return ""

    # Lowercase for processing
    text = raw.lower().strip()

    # Remove filler phrases
    words = text.split()
    filtered = []
    for word in words:
        # Remove punctuation attached to word
        clean_word = re.sub(r'[^\w$%€£¥BMKT]', '', word)
        if clean_word and clean_word not in STOPWORDS:
            filtered.append(clean_word)

    if not filtered:
        # Fallback: take first few words from original
        words = raw.split()
        filtered = [w for w in words[:3] if w.lower() not in STOPWORDS]
        if not filtered:
            filtered = words[:2]

    # Keep up to 3 most informative tokens
    # Prefer tokens that look like entities (capitalized in original) or units
    original_words = raw.split()
    informative = []
    for i, word in enumerate(original_words):
        clean = re.sub(r'[^\w$%€£¥BMKT]', '', word.lower())
        if clean and clean not in STOPWORDS:
            # Check if original had capitalization (entity)
            if word[0].isupper() and len(word) > 1:
                informative.append(word)
            elif any(re.search(pat, word) for pat in UNIT_PATTERNS):
                informative.append(word)
            else:
                informative.append(word)

    # Prefer informative tokens, fall back to filtered
    tokens = informative[:3] if informative else filtered[:3]

    # Join and Title Case
    result = " ".join(tokens)
    result = result.title()

    # Truncate on word boundary
    if len(result) > max_len:
        words = result.split()
        truncated = []
        for w in words:
            if len(" ".join(truncated + [w])) <= max_len:
                truncated.append(w)
            else:
                break
        result = " ".join(truncated) if truncated else result[:max_len].rsplit(" ", 1)[0]

    return result


def extract_key_tokens(text: str, max_tokens: int = 3) -> list[str]:
    """Extract key informative tokens from text."""
    words = text.split()
    tokens = []
    for word in text.split():
        clean = re.sub(r'[^\w$%€£¥BMKT]', '', word.lower())
        if clean and clean not in STOPWORDS and len(clean) > 1:
            tokens.append(word)
            if len(tokens) >= max_tokens:
                break
    return tokens if tokens else text.split()[:max_tokens]


import re