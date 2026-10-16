"""Stage 3 — Text Normalization.

ADR-1 Decision: spaCy lemmatization is the default normalization method.
Rationale:
  - We load spaCy en_core_web_sm for NER (Stage 4) anyway — lemmatization
    comes free on the same pipeline pass with no extra cost.
  - Lemmatization preserves meaning better than stemming:
      "developing" → "develop" ✔ (Porter: "develop" ✔)
      "universities" → "university" ✔ (Porter: "univers" ✗)
      "APIs" → "api" ✔ (Porter: "api" ✔ — but "running" → "run" vs "run" ✔)
  - Porter stemmer is available as a pluggable alternative via NormalizerConfig
    for users who need maximum speed on very large documents.

Config:
    NormalizerConfig(
        method="lemmatizer",   # or "stemmer"
        expand_acronyms=True,
        preserve_compounds=True,  # ci/cd, front-end treated as one token
        remove_stopwords=True,
    )
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── Paths ─────────────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent.parent / "data"
_ACRONYM_FILE = _DATA_DIR / "acronyms.json"


@dataclass
class NormalizerConfig:
    """Configuration for the text normalization stage."""
    method: str = "lemmatizer"          # "lemmatizer" (spaCy) | "stemmer" (Porter) | "wordnet" (WordNet)
    expand_acronyms: bool = True
    preserve_compounds: bool = True     # keep "ci/cd", "front-end" as single tokens
    remove_stopwords: bool = True
    wordnet_pos: str = "all"            # WordNet POS: "all" | "n" | "v" | "a" | "r"
    raise_on_missing_wordnet: bool = True  # Raise LookupError with clear download instructions if True


@dataclass
class NormalizerOutput:
    """Output contract for Stage 3."""
    tokens: List[str]                   # normalized token list
    lemma_map: Dict[str, str]           # original_token → lemma (for rich recs)
    expanded_text: str                  # text after acronym expansion
    method_used: str = "spacy"          # method that produced result: "spacy" | "wordnet" | "porter"
    warning: Optional[str] = None       # populated if warning or fallback occurred



# ── Stop-words ────────────────────────────────────────────────────────────────
_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "not", "no", "nor", "so", "yet", "both", "either",
    "neither", "each", "few", "more", "most", "other", "some", "such",
    "than", "too", "very", "just", "because", "as", "until", "while",
    "that", "this", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "you", "your", "he", "she", "his", "her", "who", "what",
    "which", "when", "where", "why", "how", "all", "any", "here", "there",
    "then", "once", "also", "i", "me", "my", "myself", "us", "am",
}


WORDNET_MISSING_MSG = (
    "Required NLTK WordNet data is missing. "
    "To install it, run in your terminal:\n"
    "    python -m nltk.downloader wordnet omw-1.4\n"
    "or in Python:\n"
    "    import nltk\n"
    "    nltk.download('wordnet')\n"
    "    nltk.download('omw-1.4')"
)

# ── Singleton model/resource cache ───────────────────────────────────────────
_nlp = None
_stemmer = None
_wordnet = None
_acronyms: Optional[Dict[str, str]] = None


def _load_acronyms() -> Dict[str, str]:
    global _acronyms
    if _acronyms is None:
        try:
            with open(_ACRONYM_FILE, encoding="utf-8") as f:
                _acronyms = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _acronyms = {}
    return _acronyms


def _load_spacy():
    global _nlp
    if _nlp is None:
        try:
            import spacy  # type: ignore
            _nlp = spacy.load("en_core_web_sm", disable=["ner"])  # NER loaded in Stage 4
        except (ImportError, OSError):
            _nlp = False  # mark as unavailable
    return _nlp if _nlp is not False else None


def _load_stemmer():
    global _stemmer
    if _stemmer is None:
        try:
            from nltk.stem import PorterStemmer  # type: ignore
            _stemmer = PorterStemmer()
        except ImportError:
            _stemmer = False
    return _stemmer if _stemmer is not False else None


def _load_wordnet():
    global _wordnet
    if _wordnet is not None:
        return _wordnet if _wordnet is not False else None
    try:
        from nltk.stem import WordNetLemmatizer  # type: ignore
        lem = WordNetLemmatizer()
        # Trigger corpus lookup to verify availability
        lem.lemmatize("testing", pos="v")
        _wordnet = lem
        return _wordnet
    except (ImportError, LookupError, Exception):
        _wordnet = False
        return None


# ── Public entry point ────────────────────────────────────────────────────────

def normalize(text: str, config: Optional[NormalizerConfig] = None) -> NormalizerOutput:
    """Normalize résumé or JD text into a clean token list.

    Args:
        text:   Raw plain text.
        config: Normalization configuration. Defaults to NormalizerConfig().

    Returns:
        NormalizerOutput with tokens, lemma map, expanded text, method_used, and warning.
    """
    if config is None:
        config = NormalizerConfig()

    # ── Step 1: Acronym expansion ─────────────────────────────────────────
    expanded = _expand_acronyms(text, config)

    # ── Step 2: Compound protection ───────────────────────────────────────
    # Replace / and - in compound tech terms with _ so they survive tokenization
    protected = _protect_compounds(expanded) if config.preserve_compounds else expanded

    # ── Step 3: Tokenize ──────────────────────────────────────────────────
    raw_tokens = _tokenize(protected)

    # ── Step 4: Lemmatize or stem ─────────────────────────────────────────
    method_used = "spacy"
    warning: Optional[str] = None

    if config.method in ("wordnet", "wordnet_lemmatizer"):
        tokens, lemma_map, method_used, warning = _lemmatize_wordnet(
            raw_tokens,
            pos=config.wordnet_pos,
            raise_on_missing=config.raise_on_missing_wordnet,
        )
    elif config.method in ("stemmer", "porter"):
        tokens, lemma_map = _stem(raw_tokens)
        method_used = "porter"
    else:  # default "lemmatizer" (spaCy)
        tokens, lemma_map = _lemmatize(raw_tokens)
        method_used = "spacy"

    # ── Step 5: Stopword removal ──────────────────────────────────────────
    if config.remove_stopwords:
        filtered = [(t, lemma_map.get(t, t)) for t in tokens if t not in _STOP_WORDS]
        tokens    = [t for t, _ in filtered]

    return NormalizerOutput(
        tokens=tokens,
        lemma_map=lemma_map,
        expanded_text=expanded,
        method_used=method_used,
        warning=warning,
    )


# ── Acronym expansion ─────────────────────────────────────────────────────────

def _expand_acronyms(text: str, config: NormalizerConfig) -> str:
    if not config.expand_acronyms:
        return text
    acronyms = _load_acronyms()
    # Replace whole-word acronym occurrences (case-sensitive lookup)
    for acronym, expansion in acronyms.items():
        # Match as a complete word (with optional trailing punctuation)
        pattern = r"\b" + re.escape(acronym) + r"\b"
        text = re.sub(pattern, expansion, text)
    return text


# ── Compound protection ───────────────────────────────────────────────────────

# These patterns should survive tokenization as single units
_COMPOUND_RE = re.compile(
    r"\b("
    r"[a-z][a-z0-9]*(?:[+#./\-][a-z0-9]+)+"   # ci/cd, c++, front-end, next.js
    r")\b",
    re.IGNORECASE,
)

def _protect_compounds(text: str) -> str:
    """Replace / and - in compound terms with _ to survive tokenization."""
    def _replace(m: re.Match) -> str:
        return m.group(0).replace("/", "_").replace("-", "_").replace(".", "_")
    return _COMPOUND_RE.sub(_replace, text)


# ── Tokenization ──────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer that preserves compound tokens."""
    # Extract tokens: alphanumeric + underscore (protected compounds) + + #
    raw = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_+#]*", text.lower())
    # Restore compound tokens (replace _ back to original separator context)
    return [t.replace("_", "/") if "/" in t else t for t in raw]


# ── Lemmatization via spaCy ───────────────────────────────────────────────────

def _lemmatize(tokens: List[str]) -> Tuple[List[str], Dict[str, str]]:
    nlp = _load_spacy()
    if nlp is None:
        # Fallback to identity if spaCy not available
        return tokens, {t: t for t in tokens}

    # Feed as a single space-joined string so spaCy processes it efficiently
    doc = nlp(" ".join(tokens))
    lemmas   = [token.lemma_.lower() for token in doc]
    lemma_map = {token.text: token.lemma_.lower() for token in doc}
    return lemmas, lemma_map


# ── Stemming via NLTK Porter ──────────────────────────────────────────────────

def _stem(tokens: List[str]) -> Tuple[List[str], Dict[str, str]]:
    stemmer = _load_stemmer()
    if stemmer is None:
        return tokens, {t: t for t in tokens}
    stems    = [stemmer.stem(t) for t in tokens]
    stem_map = {orig: stem for orig, stem in zip(tokens, stems)}
    return stems, stem_map


# ── Lemmatization via NLTK WordNet ───────────────────────────────────────────

def _lemmatize_wordnet(
    tokens: List[str],
    pos: str = "all",
    raise_on_missing: bool = True
) -> Tuple[List[str], Dict[str, str], str, Optional[str]]:
    """Lemmatizes tokens using WordNet.

    If WordNet data is unavailable:
      - If raise_on_missing is True: raises LookupError with clear installation instructions.
      - If raise_on_missing is False: falls back to spaCy or identity, returning an explicit
        method_used indicating the fallback (e.g. "spacy (fallback: wordnet missing)")
        along with the warning message. Never silently claims WordNet was used.
    """
    lem = _load_wordnet()
    if lem is None:
        if raise_on_missing:
            raise LookupError(WORDNET_MISSING_MSG)
        # Explicit fallback — never claim WordNet was used
        nlp = _load_spacy()
        if nlp is not None:
            lemmas, lemma_map = _lemmatize(tokens)
            return lemmas, lemma_map, "spacy (fallback: wordnet missing)", WORDNET_MISSING_MSG
        else:
            return tokens, {t: t for t in tokens}, "identity (fallback: wordnet missing)", WORDNET_MISSING_MSG

    pos_tag = (pos or "all").lower()
    valid_pos = {"n", "v", "a", "r"}

    lemmas = []
    lemma_map = {}

    for token in tokens:
        if pos_tag in valid_pos:
            lemma = lem.lemmatize(token, pos=pos_tag)
        elif pos_tag == "all":
            lemma = token
            # Try verb first, then noun, adjective, adverb
            for p in ("v", "n", "a", "r"):
                res = lem.lemmatize(token, pos=p)
                if res != token:
                    lemma = res
                    break
        else:
            lemma = lem.lemmatize(token)
        lemmas.append(lemma)
        lemma_map[token] = lemma

    return lemmas, lemma_map, "wordnet", None

