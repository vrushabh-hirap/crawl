"""Stemming and Lemmatization Tool for Crawl NLP Lab.

Supports:
- WordNet Lemmatizer (NLTK) with POS filtering and multi-POS resolution
- spaCy Lemmatizer (en_core_web_sm)
- Porter Stemmer (NLTK)
- Snowball Stemmer (NLTK)

Preserves backward compatibility, provides clear download instructions when NLTK
WordNet data is missing, and NEVER silently claims WordNet was used if a fallback
produced the result.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

WORDNET_MISSING_INSTRUCTIONS = (
    "Required NLTK WordNet data is missing. "
    "To install it, run in your terminal:\n"
    "    python -m nltk.downloader wordnet omw-1.4\n"
    "or in Python:\n"
    "    import nltk\n"
    "    nltk.download('wordnet')\n"
    "    nltk.download('omw-1.4')"
)

# ── Resource Singletons ───────────────────────────────────────────────────────
_nlp_spacy = None
_nltk_wordnet = None
_nltk_porter = None
_nltk_snowball = None


def get_spacy_lemmatizer():
    global _nlp_spacy
    if _nlp_spacy is not None:
        return _nlp_spacy if _nlp_spacy is not False else None
    try:
        import spacy  # type: ignore
        _nlp_spacy = spacy.load("en_core_web_sm", disable=["ner"])
        return _nlp_spacy
    except Exception:
        _nlp_spacy = False
        return None


def get_wordnet_lemmatizer():
    global _nltk_wordnet
    if _nltk_wordnet is not None:
        return _nltk_wordnet if _nltk_wordnet is not False else None
    try:
        from nltk.stem import WordNetLemmatizer  # type: ignore
        lem = WordNetLemmatizer()
        # Verify wordnet data is truly accessible
        lem.lemmatize("testing", pos="v")
        _nltk_wordnet = lem
        return _nltk_wordnet
    except Exception:
        _nltk_wordnet = False
        return None


def get_porter_stemmer():
    global _nltk_porter
    if _nltk_porter is not None:
        return _nltk_porter if _nltk_porter is not False else None
    try:
        from nltk.stem import PorterStemmer  # type: ignore
        _nltk_porter = PorterStemmer()
        return _nltk_porter
    except Exception:
        _nltk_porter = False
        return None


def get_snowball_stemmer():
    global _nltk_snowball
    if _nltk_snowball is not None:
        return _nltk_snowball if _nltk_snowball is not False else None
    try:
        from nltk.stem.snowball import SnowballStemmer  # type: ignore
        _nltk_snowball = SnowballStemmer("english")
        return _nltk_snowball
    except Exception:
        _nltk_snowball = False
        return None


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def tokenize_words(text: str) -> List[str]:
    """Extracts words and compound terms while discarding isolated punctuation."""
    if not text:
        return []
    # Match words, compound terms like 'ci/cd', 'front-end', 'c++', 'k8s'
    pattern = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9_+#/'-]*\b")
    return pattern.findall(text)


# ── WordNet Lemmatization Logic ───────────────────────────────────────────────

def lemmatize_wordnet_word(lemmatizer: Any, word: str, pos: str = "all") -> Tuple[str, str]:
    """Lemmatizes a single word using WordNet.

    Returns:
        (lemma, pos_used)
    """
    valid_pos = {"n", "v", "a", "r"}
    pos_tag = (pos or "all").lower()

    if pos_tag in valid_pos:
        lemma = lemmatizer.lemmatize(word.lower(), pos=pos_tag)
        return lemma, pos_tag

    # If "all" or unrecognized, check verbs first, then nouns, adjectives, adverbs
    w_lower = word.lower()
    for p in ("v", "n", "a", "r"):
        lemma = lemmatizer.lemmatize(w_lower, pos=p)
        if lemma != w_lower:
            return lemma, p

    return lemmatizer.lemmatize(w_lower), "n"


# ── Public Function ───────────────────────────────────────────────────────────

def stem_and_lemmatize(text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Normalizes natural-language text using WordNet, spaCy, or stemmers.

    Args:
        text: Input string to process.
        options: Optional dict supporting:
            - 'method': "wordnet" (default), "spacy", "porter", "snowball"
            - 'pos': WordNet POS filter: "all" (default), "n", "v", "a", "r"
            - 'fallback_on_missing': bool (default False). If True and WordNet data is
              missing, falls back to another method while explicitly labeling method_used
              and warning. If False, raises LookupError with clear instructions.

    Returns:
        Dictionary containing:
            - method_requested: Requested algorithm string
            - method_used: Actual algorithm that produced the output (e.g. "wordnet",
              "spacy", or "spacy (fallback: wordnet missing)")
            - is_fallback: bool indicating whether fallback occurred
            - warning: Warning message with download instructions if fallback occurred
            - algorithm: Description of algorithm used
            - tokens: List of transformed token strings
            - items: List of dicts with original, normalized, changed, and pos
            - stats: Transformation counts and rate
    """
    if options is None:
        options = {}

    method_requested = str(options.get("method", "wordnet")).lower()
    pos_opt = str(options.get("pos", "all")).lower()
    fallback_on_missing = bool(options.get("fallback_on_missing", False))

    raw_tokens = tokenize_words(text)
    if not raw_tokens:
        return {
            "method_requested": method_requested,
            "method_used": method_requested,
            "is_fallback": False,
            "warning": None,
            "algorithm": f"No tokens extracted for method '{method_requested}'",
            "tokens": [],
            "items": [],
            "stats": {
                "total_tokens": 0,
                "transformed_tokens": 0,
                "unchanged_tokens": 0,
                "transformation_rate": 0.0,
            },
        }

    method_used = method_requested
    is_fallback = False
    warning: Optional[str] = None
    algorithm = ""
    items: List[Dict[str, Any]] = []

    # 1. WordNet Lemmatization
    if method_requested in ("wordnet", "wordnet_lemmatizer"):
        wn_lem = get_wordnet_lemmatizer()
        if wn_lem is None:
            if not fallback_on_missing:
                raise LookupError(WORDNET_MISSING_INSTRUCTIONS)

            # Explicit fallback
            warning = WORDNET_MISSING_INSTRUCTIONS
            is_fallback = True
            spacy_nlp = get_spacy_lemmatizer()
            if spacy_nlp is not None:
                method_used = "spacy (fallback: wordnet missing)"
                algorithm = "spaCy lemmatizer (fallback due to missing NLTK WordNet data)"
                doc = spacy_nlp(" ".join(raw_tokens))
                for orig, sp_tok in zip(raw_tokens, doc):
                    norm = sp_tok.lemma_.lower()
                    items.append({
                        "original": orig,
                        "normalized": norm,
                        "changed": orig.lower() != norm.lower(),
                        "pos": sp_tok.pos_.lower(),
                    })
            else:
                porter = get_porter_stemmer()
                if porter is not None:
                    method_used = "porter (fallback: wordnet missing)"
                    algorithm = "Porter stemmer (fallback due to missing NLTK WordNet data)"
                    for orig in raw_tokens:
                        norm = porter.stem(orig.lower())
                        items.append({
                            "original": orig,
                            "normalized": norm,
                            "changed": orig.lower() != norm.lower(),
                            "pos": "stem",
                        })
                else:
                    method_used = "identity (fallback: wordnet missing)"
                    algorithm = "Identity tokens (no morphological models available)"
                    for orig in raw_tokens:
                        items.append({
                            "original": orig,
                            "normalized": orig.lower(),
                            "changed": False,
                            "pos": "unknown",
                        })
        else:
            algorithm = f"WordNet Lemmatizer (NLTK Morphy, POS='{pos_opt}')"
            for orig in raw_tokens:
                norm, pos_used = lemmatize_wordnet_word(wn_lem, orig, pos=pos_opt)
                items.append({
                    "original": orig,
                    "normalized": norm,
                    "changed": orig.lower() != norm.lower(),
                    "pos": pos_used,
                })

    # 2. spaCy Lemmatization
    elif method_requested in ("spacy", "lemmatizer"):
        spacy_nlp = get_spacy_lemmatizer()
        if spacy_nlp is None:
            # Fall back to Porter
            warning = "spaCy model 'en_core_web_sm' is unavailable; fell back to Porter stemmer."
            is_fallback = True
            method_used = "porter (fallback: spacy unavailable)"
            algorithm = "Porter stemmer (fallback)"
            porter = get_porter_stemmer()
            for orig in raw_tokens:
                norm = porter.stem(orig.lower()) if porter else orig.lower()
                items.append({
                    "original": orig,
                    "normalized": norm,
                    "changed": orig.lower() != norm.lower(),
                    "pos": "stem",
                })
        else:
            algorithm = "spaCy Lemmatizer (en_core_web_sm)"
            doc = spacy_nlp(" ".join(raw_tokens))
            for orig, sp_tok in zip(raw_tokens, doc):
                norm = sp_tok.lemma_.lower()
                items.append({
                    "original": orig,
                    "normalized": norm,
                    "changed": orig.lower() != norm.lower(),
                    "pos": sp_tok.pos_.lower(),
                })

    # 3. Porter Stemmer
    elif method_requested in ("porter", "stemmer"):
        porter = get_porter_stemmer()
        if porter is None:
            raise RuntimeError("NLTK PorterStemmer could not be loaded.")
        algorithm = "Porter Stemmer (NLTK algorithmic suffix stripping)"
        for orig in raw_tokens:
            norm = porter.stem(orig.lower())
            items.append({
                "original": orig,
                "normalized": norm,
                "changed": orig.lower() != norm.lower(),
                "pos": "stem",
            })

    # 4. Snowball Stemmer
    elif method_requested in ("snowball", "snowball_stemmer"):
        snowball = get_snowball_stemmer()
        if snowball is None:
            raise RuntimeError("NLTK SnowballStemmer could not be loaded.")
        algorithm = "Snowball Stemmer (Porter2 English)"
        for orig in raw_tokens:
            norm = snowball.stem(orig.lower())
            items.append({
                "original": orig,
                "normalized": norm,
                "changed": orig.lower() != norm.lower(),
                "pos": "stem",
            })

    else:
        raise ValueError(
            f"Unsupported normalization method '{method_requested}'. "
            "Supported methods: ['wordnet', 'spacy', 'porter', 'snowball']"
        )

    # Compute summary stats
    tokens = [item["normalized"] for item in items]
    total_tokens = len(items)
    transformed_count = sum(1 for item in items if item["changed"])
    unchanged_count = total_tokens - transformed_count
    rate = round(transformed_count / total_tokens, 3) if total_tokens > 0 else 0.0

    return {
        "method_requested": method_requested,
        "method_used": method_used,
        "is_fallback": is_fallback,
        "warning": warning,
        "algorithm": algorithm,
        "tokens": tokens,
        "items": items,
        "stats": {
            "total_tokens": total_tokens,
            "transformed_tokens": transformed_count,
            "unchanged_tokens": unchanged_count,
            "transformation_rate": rate,
        },
    }
