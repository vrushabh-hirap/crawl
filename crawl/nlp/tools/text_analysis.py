"""Text Analysis tool: language detection, word count, sentence count, and tokenization.

Designed specifically for natural language processing, completely decoupled from
Crawl's HTML markup tokenizer. Supports multilingual language detection with clear
uncertainty indicators, empty input handling, and standard library fallbacks.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

# ISO 639-1 code to Language Name mapping
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "da": "Danish",
    "sv": "Swedish",
    "no": "Norwegian",
    "fi": "Finnish",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "tr": "Turkish",
    "ar": "Arabic",
    "id": "Indonesian",
    "ca": "Catalan",
    "eu": "Basque",
    "sl": "Slovene",
    "ne": "Nepali",
    "uz": "Uzbek",
    "az": "Azerbaijani",
    "bn": "Bengali",
    "ta": "Tamil",
    "he": "Hebrew",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "unknown": "Unknown",
}

# Mapping between NLTK stopword fileids and ISO codes
_NLTK_LANG_MAP: Dict[str, str] = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "dutch": "nl",
    "russian": "ru",
    "danish": "da",
    "swedish": "sv",
    "norwegian": "no",
    "finnish": "fi",
    "greek": "el",
    "romanian": "ro",
    "hungarian": "hu",
    "turkish": "tr",
    "arabic": "ar",
    "indonesian": "id",
    "catalan": "ca",
    "basque": "eu",
    "slovene": "sl",
    "nepali": "ne",
    "uzbek": "uz",
    "azerbaijani": "az",
    "bengali": "bn",
    "tamil": "ta",
    "hebrew": "he",
    "chinese": "zh",
}

# High-frequency grammatical markers for core languages (fallback if NLTK corpus unavailable)
_CORE_FALLBACK_STOPWORDS: Dict[str, Set[str]] = {
    "en": {
        "the", "and", "of", "to", "in", "is", "that", "for", "with", "it",
        "as", "was", "on", "are", "be", "this", "have", "from", "or", "an",
        "by", "not", "we", "you", "can", "at", "but", "all", "they", "our",
        "what", "which", "will", "more", "about", "if", "out", "so", "up",
    },
    "es": {
        "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las",
        "por", "un", "para", "con", "no", "una", "su", "al", "lo", "como",
        "mas", "más", "pero", "sus", "le", "ya", "o", "este", "si", "sí",
        "porque", "esta", "son", "entre", "cuando", "muy", "sin", "sobre",
    },
    "fr": {
        "de", "la", "le", "et", "les", "des", "en", "un", "du", "une", "est",
        "pour", "dans", "a", "à", "au", "qui", "ce", "sur", "pas", "plus",
        "par", "avec", "sont", "il", "se", "ne", "mais", "ou", "nous", "vous",
        "comme", "leur", "aussi", "son", "sa", "ses", "ont", "aux",
    },
    "de": {
        "der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich",
        "des", "auf", "fur", "für", "ist", "im", "dem", "nicht", "ein", "eine",
        "als", "auch", "es", "an", "werden", "aus", "er", "hat", "dass",
        "sie", "nach", "bei", "einer", "wir", "zur", "oder", "wenn",
    },
    "it": {
        "di", "e", "il", "che", "la", "a", "in", "per", "un", "del", "della",
        "da", "le", "con", "si", "non", "una", "sono", "dei", "delle", "ed",
        "al", "lo", "ha", "ci", "come", "anche", "piu", "più", "questo", "gli",
    },
    "pt": {
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com",
        "nao", "não", "uma", "os", "no", "se", "na", "por", "mais", "as",
        "dos", "como", "mas", "ao", "ele", "das", "tem", "seu", "sua",
    },
    "nl": {
        "de", "van", "en", "het", "in", "is", "een", "op", "te", "dat", "die",
        "voor", "met", "zijn", "niet", "aan", "ook", "als", "om", "bij", "er",
        "maar", "naar", "uit", "door", "over", "ze", "we", "al",
    },
    "ru": {
        "и", "в", "не", "на", "я", "что", "тот", "быть", "с", "он", "а",
        "как", "это", "по", "этот", "к", "но", "они", "мы", "из", "у",
        "который", "то", "за", "свое", "же", "вы", "так", "его",
    },
}

# Cache for loaded stopword dictionaries
_LOADED_STOPWORDS: Optional[Dict[str, Set[str]]] = None


def _get_stopwords_dict() -> Dict[str, Set[str]]:
    """Loads stopwords from NLTK if available, falling back to core dictionaries."""
    global _LOADED_STOPWORDS
    if _LOADED_STOPWORDS is not None:
        return _LOADED_STOPWORDS

    loaded: Dict[str, Set[str]] = {k: set(v) for k, v in _CORE_FALLBACK_STOPWORDS.items()}
    try:
        from nltk.corpus import stopwords  # type: ignore
        for nltk_lang in stopwords.fileids():
            iso_code = _NLTK_LANG_MAP.get(nltk_lang.lower())
            if iso_code:
                try:
                    words = set(stopwords.words(nltk_lang))
                    if iso_code in loaded:
                        loaded[iso_code].update(words)
                    else:
                        loaded[iso_code] = words
                except Exception:
                    continue
    except Exception:
        pass

    _LOADED_STOPWORDS = loaded
    return _LOADED_STOPWORDS


# ─────────────────────────────────────────────────────────────────────────────
# Language Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_language(text: str) -> Dict[str, Any]:
    """Detects document language, with clear indications when detection is uncertain or unavailable.

    Args:
        text: Input string.

    Returns:
        Dictionary with:
          - language: ISO 639-1 code (e.g. 'en') or 'unknown'
          - language_name: Human readable name (e.g. 'English') or 'Unknown'
          - is_uncertain: Boolean indicating whether detection was uncertain
          - confidence: Confidence score between 0.0 and 1.0
          - status: 'detected' | 'uncertain' | 'unavailable' | 'unsupported'
          - message: Explanation when detection is uncertain or unavailable
    """
    # 1. Empty or whitespace text
    if not text or not text.strip():
        return {
            "language": "unknown",
            "language_name": "Unknown",
            "is_uncertain": True,
            "confidence": 0.0,
            "status": "unavailable",
            "message": "Input text is empty. Language detection unavailable."
        }

    clean_text = text.strip()

    # 2. Distinct non-Latin script identification
    if re.search(r"[\u4e00-\u9fff]", clean_text):
        return {
            "language": "zh",
            "language_name": "Chinese",
            "is_uncertain": False,
            "confidence": 0.98,
            "status": "detected",
            "message": None
        }
    if re.search(r"[\u3040-\u30ff]", clean_text):
        return {
            "language": "ja",
            "language_name": "Japanese",
            "is_uncertain": False,
            "confidence": 0.98,
            "status": "detected",
            "message": None
        }
    if re.search(r"[\uac00-\ud7af]", clean_text):
        return {
            "language": "ko",
            "language_name": "Korean",
            "is_uncertain": False,
            "confidence": 0.98,
            "status": "detected",
            "message": None
        }
    if re.search(r"[\u0400-\u04ff]", clean_text):
        return {
            "language": "ru",
            "language_name": "Russian",
            "is_uncertain": False,
            "confidence": 0.96,
            "status": "detected",
            "message": None
        }
    if re.search(r"[\u0370-\u03ff]", clean_text):
        return {
            "language": "el",
            "language_name": "Greek",
            "is_uncertain": False,
            "confidence": 0.96,
            "status": "detected",
            "message": None
        }
    if re.search(r"[\u0600-\u06ff]", clean_text):
        return {
            "language": "ar",
            "language_name": "Arabic",
            "is_uncertain": False,
            "confidence": 0.96,
            "status": "detected",
            "message": None
        }
    if re.search(r"[\u0590-\u05ff]", clean_text):
        return {
            "language": "he",
            "language_name": "Hebrew",
            "is_uncertain": False,
            "confidence": 0.96,
            "status": "detected",
            "message": None
        }
    if re.search(r"[\u0900-\u097f]", clean_text):
        return {
            "language": "hi",
            "language_name": "Hindi",
            "is_uncertain": False,
            "confidence": 0.95,
            "status": "detected",
            "message": None
        }

    # 3. Extract word tokens for lexical profile analysis
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z\u00C0-\u017F]+\b", clean_text)]
    if not words:
        return {
            "language": "unknown",
            "language_name": "Unknown",
            "is_uncertain": True,
            "confidence": 0.0,
            "status": "unavailable",
            "message": "No recognizable word tokens found in text."
        }

    stopwords_map = _get_stopwords_dict()

    # 4. Short text handling (under 3 words)
    if len(words) < 3:
        # Check if single word matches high-frequency stopword
        for code, sw in stopwords_map.items():
            if any(w in sw for w in words):
                lang_name = LANGUAGE_NAMES.get(code, code.title())
                return {
                    "language": code,
                    "language_name": lang_name,
                    "is_uncertain": True,
                    "confidence": 0.35,
                    "status": "uncertain",
                    "message": f"Text is very short ({len(words)} word{'s' if len(words) > 1 else ''}); classification is tentative."
                }
        return {
            "language": "unknown",
            "language_name": "Unknown",
            "is_uncertain": True,
            "confidence": 0.0,
            "status": "uncertain",
            "message": "Text is too short for reliable language identification."
        }

    # 5. Score against linguistic stopword sets
    scores: Dict[str, int] = {}
    for code, sw in stopwords_map.items():
        hits = sum(1 for w in words if w in sw)
        if hits > 0:
            scores[code] = hits

    # 6. Check if no recognized language markers matched (unsupported language or gibberish)
    if not scores:
        return {
            "language": "unknown",
            "language_name": "Unknown",
            "is_uncertain": True,
            "confidence": 0.0,
            "status": "unsupported",
            "message": "No recognized language markers detected. The language may be unsupported or non-lexical."
        }

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_code, top_hits = sorted_scores[0]
    runner_hits = sorted_scores[1][1] if len(sorted_scores) > 1 else 0

    # 7. Check ambiguous ties
    if top_hits == runner_hits:
        runner_code = sorted_scores[1][0]
        top_name = LANGUAGE_NAMES.get(top_code, top_code.title())
        runner_name = LANGUAGE_NAMES.get(runner_code, runner_code.title())
        return {
            "language": "unknown",
            "language_name": "Unknown",
            "is_uncertain": True,
            "confidence": 0.40,
            "status": "uncertain",
            "message": f"Language detection is ambiguous between {top_name} and {runner_name}."
        }

    # 8. High confidence detection
    diff = top_hits - runner_hits
    confidence = min(0.99, round(0.55 + 0.45 * (diff / (top_hits + 0.5)), 2))
    lang_name = LANGUAGE_NAMES.get(top_code, top_code.title())

    return {
        "language": top_code,
        "language_name": lang_name,
        "is_uncertain": False,
        "confidence": confidence,
        "status": "detected",
        "message": None
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sentence Splitting & Tokenization (Pure Natural Language, decoupled from HTML)
# ─────────────────────────────────────────────────────────────────────────────

# Common abbreviations to prevent premature sentence boundary splitting
_COMMON_ABBREVIATIONS = (
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.",
    "e.g.", "i.e.", "u.s.", "inc.", "ltd.", "corp.", "co.", "approx.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "oct.", "nov.", "dec."
)


def split_sentences(text: str) -> List[str]:
    """Splits natural language text into sentences using abbreviation-aware rules.

    Args:
        text: Input string.

    Returns:
        List of trimmed sentence strings.
    """
    if not text or not text.strip():
        return []

    # Try spaCy sentence segmenter first if loaded
    try:
        import spacy  # type: ignore
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
        doc = nlp(text.strip())
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        if sents:
            return sents
    except Exception:
        pass

    # High-accuracy fallback: Mask abbreviations and decimals before splitting
    masked = text.strip()
    placeholder_map: Dict[str, str] = {}

    for i, abbr in enumerate(_COMMON_ABBREVIATIONS):
        pattern = re.compile(re.escape(abbr), re.IGNORECASE)
        for match in pattern.finditer(masked):
            token = match.group(0)
            placeholder = f"__ABBR_{i}_{len(placeholder_map)}__"
            placeholder_map[placeholder] = token
            masked = masked.replace(token, placeholder, 1)

    # Protect decimals like 3.14 or v1.0
    decimal_matches = list(re.finditer(r"(\d+)\.(\d+)", masked))
    for i, m in enumerate(decimal_matches):
        orig = m.group(0)
        ph = f"__DEC_{i}__"
        placeholder_map[ph] = orig
        masked = masked.replace(orig, ph, 1)

    # Split on sentence boundary punctuation (. ! ?) followed by whitespace or end
    parts = re.split(r"([.!?]+(?:\s+|$))", masked)
    sentences: List[str] = []
    current_acc = ""

    for p in parts:
        current_acc += p
        if re.search(r"[.!?]", p):
            trimmed = current_acc.strip()
            if trimmed:
                sentences.append(trimmed)
            current_acc = ""

    if current_acc.strip():
        sentences.append(current_acc.strip())

    # Restore placeholders
    restored: List[str] = []
    for s in sentences:
        for ph, orig in placeholder_map.items():
            s = s.replace(ph, orig)
        if s.strip():
            restored.append(s.strip())

    return restored if restored else ([text.strip()] if text.strip() else [])


def tokenize_text(
    text: str,
    lowercase: bool = False,
    strip_punctuation: bool = False
) -> List[str]:
    """Tokenizes natural language text into words and punctuation tokens.

    Preserves contractions (don't), hyphenated compounds (state-of-the-art),
    and numbers while properly tokenizing trailing punctuation.

    Args:
        text: Input text string.
        lowercase: Whether to convert all tokens to lowercase.
        strip_punctuation: Whether to exclude purely punctuation tokens.

    Returns:
        List of string tokens.
    """
    if not text or not text.strip():
        return []

    # Regex matching:
    # 1. Contractions and hyphenated compounds: don't, state-of-the-art
    # 2. Decimals and numbers: 3.14, 1,000
    # 3. Unicode words: characters including accented letters
    # 4. Punctuation characters
    token_pattern = re.compile(
        r"[A-Za-z0-9\u00C0-\u017F]+(?:['’\-_][A-Za-z0-9\u00C0-\u017F]+)+|\d+(?:[.,]\d+)?|[\w\u00C0-\u017F]+|[^\w\s]"
    )

    raw_tokens = token_pattern.findall(text)

    tokens: List[str] = []
    for tok in raw_tokens:
        tok_clean = tok.strip()
        if not tok_clean:
            continue

        is_punct = bool(re.fullmatch(r"[^\w\s]+", tok_clean))
        if strip_punctuation and is_punct:
            continue

        if lowercase:
            tok_clean = tok_clean.lower()

        tokens.append(tok_clean)

    return tokens


def count_words(text: str) -> int:
    """Calculates the total number of natural language words in the text."""
    if not text or not text.strip():
        return 0
    # Match alphanumeric words, ignoring pure punctuation
    words = re.findall(r"\b[A-Za-z0-9\u00C0-\u017F]+(?:['’\-_][A-Za-z0-9\u00C0-\u017F]+)*\b", text)
    return len(words)


def count_sentences(text: str) -> int:
    """Calculates the total number of sentences in the text."""
    if not text or not text.strip():
        return 0
    return len(split_sentences(text))


# ─────────────────────────────────────────────────────────────────────────────
# Main Text Analysis Interface
# ─────────────────────────────────────────────────────────────────────────────

def analyze_text(text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Analyzes text for language, word count, sentence count, and token stream.

    Handles empty input, unsupported languages, and option parameters gracefully.

    Args:
        text: Input string.
        options: Optional dict with 'lowercase' and 'strip_punctuation' flags.

    Returns:
        Structured dictionary matching the Text Analysis specification:
          - language: ISO 639-1 code ('en', 'es', etc.) or 'unknown'
          - language_name: Human readable language name
          - is_uncertain: Boolean flag indicating if language detection was uncertain
          - language_confidence: Confidence score between 0.0 and 1.0
          - language_status: 'detected' | 'uncertain' | 'unavailable' | 'unsupported'
          - language_message: Detailed uncertainty explanation or None
          - word_count: Total word count
          - sentence_count: Total sentence count
          - character_count: Total raw characters
          - tokens: List of extracted tokens
          - tokens_count: Total number of tokens
          - sentences: List of sentence strings
    """
    raw_options = options if isinstance(options, dict) else {}
    lowercase = bool(raw_options.get("lowercase", False))
    strip_punctuation = bool(raw_options.get("strip_punctuation", False))

    # Graceful handling of empty or blank text
    if not text or not text.strip():
        lang_res = detect_language(text or "")
        return {
            "language": lang_res["language"],
            "language_name": lang_res["language_name"],
            "is_uncertain": lang_res["is_uncertain"],
            "language_confidence": lang_res["confidence"],
            "language_status": lang_res["status"],
            "language_message": lang_res["message"],
            "word_count": 0,
            "sentence_count": 0,
            "character_count": len(text) if text else 0,
            "tokens": [],
            "tokens_count": 0,
            "sentences": [],
        }

    # 1. Language detection with uncertainty indicators
    lang_res = detect_language(text)

    # 2. Sentences
    sentences = split_sentences(text)
    sentence_count = len(sentences)

    # 3. Words
    word_count = count_words(text)

    # 4. Tokens
    tokens = tokenize_text(text, lowercase=lowercase, strip_punctuation=strip_punctuation)

    return {
        "language": lang_res["language"],
        "language_name": lang_res["language_name"],
        "is_uncertain": lang_res["is_uncertain"],
        "language_confidence": lang_res["confidence"],
        "language_status": lang_res["status"],
        "language_message": lang_res["message"],
        "word_count": word_count,
        "sentence_count": sentence_count,
        "character_count": len(text),
        "tokens": tokens,
        "tokens_count": len(tokens),
        "sentences": sentences,
    }
