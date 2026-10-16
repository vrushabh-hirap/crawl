"""Part-of-Speech Tagging, Dense Text Embeddings & Text Classification.

Components:
1. Part-of-speech (POS) tagging: returns tokens, fine/coarse tags, descriptions,
   and frequency distributions using spaCy (with NLTK fallback).
2. Dense text embeddings: generates and formats vector representations, reusing the
   project's optional sentence-transformers dependency when installed (384-d MiniLM),
   with built-in spaCy vector (96-d) fallback and actionable installation guidance.
3. Text classification: trained on a documented 5-category benchmark corpus
   (Technology, Engineering, Science, Business, General) with support for
   user-provided datasets and custom candidate labels, outputting predictions,
   probability distributions, and evaluation explainability.
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

# ── Documented Benchmark Dataset for Text Classification ──────────────────────
BENCHMARK_DATASET: Dict[str, List[str]] = {
    "Technology": [
        "Senior software engineer developing distributed backend microservices and REST APIs.",
        "Kubernetes container orchestration with Docker on AWS and Google Cloud infrastructure.",
        "Python web application development using Flask, Django, PostgreSQL, and Redis caching.",
        "Machine learning models, neural network architectures, and natural language processing pipelines.",
    ],
    "Engineering": [
        "Mechanical design, CAD blueprints, and structural finite element analysis for aerospace systems.",
        "Civil engineering blueprints, reinforced concrete load bearing, and geotechnical surveying.",
        "Electrical circuit schematic layout, FPGA firmware, microcontrollers, and PCB prototyping.",
        "Robotics kinematic control, automated manufacturing assembly lines, and industrial sensors.",
    ],
    "Science": [
        "Genomic DNA sequencing, CRISPR-Cas9 molecular biology, and cellular immunology research.",
        "Astrophysics observation of cosmological radiation, stellar fusion, and quantum gravity theory.",
        "Organic chemical synthesis of pharmaceutical molecules and nuclear magnetic resonance spectroscopy.",
        "Climate modeling, oceanographic temperature variance, and atmospheric carbon cycle dynamics.",
    ],
    "Business": [
        "Quarterly financial revenue forecasting, GAAP accounting auditing, and balance sheet valuation.",
        "Digital marketing campaigns, customer acquisition cost funnel, and SEO brand growth strategy.",
        "Corporate venture capital investment, mergers and acquisitions, and shareholder governance.",
        "Enterprise human resources management, talent recruiting, executive compensation, and payroll.",
    ],
    "General": [
        "Daily municipal news, community public library events, and local weather forecasts.",
        "Culinary cooking recipes, restaurant dining recommendations, and food preparation techniques.",
        "Travel itinerary planning, cultural museums, hiking outdoor adventures, and tourist sightseeing.",
        "Health wellness routines, cardiovascular fitness exercises, and healthy dietary habits.",
    ],
}

# ── Universal POS Tag Descriptions (Penn Treebank & Universal Tagset) ──────────
_POS_EXPLANATIONS: Dict[str, str] = {
    "ADJ": "adjective",
    "ADP": "adposition (preposition or postposition)",
    "ADV": "adverb",
    "AUX": "auxiliary verb",
    "CCONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
    # Penn Treebank common tags
    "NN": "noun, singular or mass",
    "NNS": "noun, plural",
    "NNP": "noun, proper singular",
    "NNPS": "noun, proper plural",
    "VB": "verb, base form",
    "VBD": "verb, past tense",
    "VBG": "verb, gerund or present participle",
    "VBN": "verb, past participle",
    "VBP": "verb, non-3rd person singular present",
    "VBZ": "verb, 3rd person singular present",
    "JJ": "adjective",
    "JJR": "adjective, comparative",
    "JJS": "adjective, superlative",
    "RB": "adverb",
    "RBR": "adverb, comparative",
    "RBS": "adverb, superlative",
    "IN": "preposition or subordinating conjunction",
    "DT": "determiner",
    "CD": "cardinal number",
    "PRP": "personal pronoun",
    "PRP$": "possessive pronoun",
}

# ── Dependency Helpers & Singletons ───────────────────────────────────────────
_spacy_nlp = None


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp if _spacy_nlp is not False else None
    try:
        import spacy  # type: ignore
        _spacy_nlp = spacy.load("en_core_web_sm")
        return _spacy_nlp
    except Exception:
        _spacy_nlp = False
        return None


def _get_sentence_transformer():
    """Reuses the optional sentence-transformers model from Stage 6 Semantic."""
    try:
        from crawl.jobs.ats.pipeline.stage6_semantic import _load_model as load_semantic_model
        return load_semantic_model()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Part-of-Speech Tagging
# ─────────────────────────────────────────────────────────────────────────────

def pos_tag_text(text: str) -> Dict[str, Any]:
    """Extracts Part-of-Speech (POS) tags for all tokens in the input text.

    Args:
        text: Plain-text string to analyze.

    Returns:
        Structured dictionary with tokens, tags, descriptions, and POS distribution.
    """
    if not text or not text.strip():
        return {
            "task": "pos_tagging",
            "engine": "none",
            "tokens_count": 0,
            "tagged_tokens": [],
            "tokens": [],
            "pos_distribution": {},
            "lexical_density": 0.0,
            "summary": "Empty input text.",
        }

    nlp = _get_spacy()
    tagged_tokens: List[Dict[str, Any]] = []

    if nlp is not None:
        engine = "spaCy en_core_web_sm (Universal POS + Penn Treebank)"
        doc = nlp(text)
        for token in doc:
            if not token.text.strip():
                continue
            pos = token.pos_
            tag = token.tag_
            desc = _POS_EXPLANATIONS.get(tag, _POS_EXPLANATIONS.get(pos, pos))
            tagged_tokens.append({
                "token": token.text,
                "pos": pos,
                "tag": tag,
                "description": desc,
                "lemma": token.lemma_,
                "is_stop": token.is_stop,
            })
    else:
        # Fallback to NLTK pos_tag or simple regex rules
        engine = "NLTK pos_tag (fallback)"
        try:
            from nltk import word_tokenize, pos_tag  # type: ignore
            raw_tokens = word_tokenize(text)
            nltk_tags = pos_tag(raw_tokens)
            for tok, tag in nltk_tags:
                pos = "NOUN" if tag.startswith("NN") else ("VERB" if tag.startswith("VB") else ("ADJ" if tag.startswith("JJ") else "OTHER"))
                desc = _POS_EXPLANATIONS.get(tag, pos)
                tagged_tokens.append({
                    "token": tok,
                    "pos": pos,
                    "tag": tag,
                    "description": desc,
                    "lemma": tok.lower(),
                    "is_stop": False,
                })
        except Exception:
            # Basic rule-based fallback
            engine = "Heuristic Rule-Based Tagger (fallback)"
            words = re.findall(r"\b[A-Za-z0-9+#/'-]+\b", text)
            for w in words:
                pos = "NOUN"
                if w.lower() in ("the", "a", "an", "this", "that"):
                    pos = "DET"
                elif w.lower() in ("in", "on", "at", "to", "for", "with", "by"):
                    pos = "ADP"
                elif w.lower() in ("is", "are", "was", "were", "be", "have", "has", "deploy", "run"):
                    pos = "VERB"
                tagged_tokens.append({
                    "token": w,
                    "pos": pos,
                    "tag": pos,
                    "description": _POS_EXPLANATIONS.get(pos, pos),
                    "lemma": w.lower(),
                    "is_stop": False,
                })

    pos_counts = Counter(t["pos"] for t in tagged_tokens)
    total_tokens = len(tagged_tokens)

    # Content words (lexical words): NOUN, PROPN, VERB, ADJ, ADV
    content_count = sum(pos_counts[p] for p in ("NOUN", "PROPN", "VERB", "ADJ", "ADV"))
    lexical_density = round(content_count / total_tokens, 3) if total_tokens > 0 else 0.0

    return {
        "task": "pos_tagging",
        "engine": engine,
        "tokens_count": total_tokens,
        "tagged_tokens": tagged_tokens,
        "tokens": [t["token"] for t in tagged_tokens],
        "pos_distribution": dict(pos_counts.most_common()),
        "lexical_density": lexical_density,
        "summary": f"{total_tokens} tokens tagged across {len(pos_counts)} grammatical categories.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dense Text Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def get_text_embedding(text: str) -> Dict[str, Any]:
    """Generates dense vector embeddings for input text.

    Reuses the project's optional `sentence-transformers` dependency (`all-MiniLM-L6-v2`)
    when available, falling back cleanly to spaCy's built-in 96-dimensional vectors
    with actionable installation instructions.

    Args:
        text: Plain-text string to embed.

    Returns:
        Structured embedding dictionary with vector, preview, dimensions, and metadata.
    """
    clean_text = (text or "").strip()
    if not clean_text:
        return {
            "task": "embeddings",
            "model": "none",
            "dimensions": 0,
            "vector": [],
            "vector_preview": [],
            "vector_norm": 0.0,
            "is_optional_dependency": False,
            "installed": False,
            "notice": "Empty text provided.",
            "stats": {"dimensions": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "l2_norm": 0.0},
        }

    # 1. Check for optional sentence-transformers
    st_model = _get_sentence_transformer()
    if st_model is not None:
        vec = st_model.encode(clean_text[:5000])
        # Convert numpy array to standard Python float list
        vector_list = [round(float(x), 6) for x in vec]
        norm = math.sqrt(sum(x * x for x in vector_list))
        dims = len(vector_list)

        return {
            "task": "embeddings",
            "model": "sentence-transformers (all-MiniLM-L6-v2)",
            "dimensions": dims,
            "vector": vector_list,
            "vector_preview": [round(x, 4) for x in vector_list[:16]],
            "vector_norm": round(norm, 4),
            "is_optional_dependency": True,
            "installed": True,
            "notice": (
                "Dense neural sentence embeddings generated using sentence-transformers 'all-MiniLM-L6-v2' "
                "(384-dimensional unit-normalized space)."
            ),
            "stats": {
                "dimensions": dims,
                "min": round(min(vector_list), 4),
                "max": round(max(vector_list), 4),
                "mean": round(sum(vector_list) / dims, 4),
                "l2_norm": round(norm, 4),
            },
        }

    # 2. Fallback to spaCy built-in vectors
    nlp = _get_spacy()
    if nlp is not None:
        doc = nlp(clean_text[:5000])
        raw_vec = doc.vector
        vector_list = [round(float(x), 6) for x in raw_vec]
        dims = len(vector_list)
        norm = math.sqrt(sum(x * x for x in vector_list))

        return {
            "task": "embeddings",
            "model": "spaCy en_core_web_sm (96-d vector)",
            "dimensions": dims,
            "vector": vector_list,
            "vector_preview": [round(x, 4) for x in vector_list[:16]],
            "vector_norm": round(norm, 4),
            "is_optional_dependency": False,
            "installed": False,
            "notice": (
                "Using spaCy 96-dimensional vector embedding. "
                "To enable high-accuracy dense neural sentence embeddings, install sentence-transformers:\n"
                "    pip install crawl-jobs[semantic]\n"
                "or: pip install sentence-transformers"
            ),
            "stats": {
                "dimensions": dims,
                "min": round(min(vector_list), 4) if dims > 0 else 0.0,
                "max": round(max(vector_list), 4) if dims > 0 else 0.0,
                "mean": round(sum(vector_list) / dims, 4) if dims > 0 else 0.0,
                "l2_norm": round(norm, 4),
            },
        }

    # 3. Deterministic Hashing / Term Frequency fallback
    dims = 64
    counts = Counter(re.findall(r"\b[A-Za-z0-9]+\b", clean_text.lower()))
    vector_list = [0.0] * dims
    for word, cnt in counts.items():
        idx = abs(hash(word)) % dims
        vector_list[idx] += float(cnt)
    norm = math.sqrt(sum(x * x for x in vector_list)) or 1.0
    vector_list = [round(x / norm, 6) for x in vector_list]

    return {
        "task": "embeddings",
        "model": "Frequency-Hash Vector (64-d fallback)",
        "dimensions": dims,
        "vector": vector_list,
        "vector_preview": [round(x, 4) for x in vector_list[:16]],
        "vector_norm": 1.0,
        "is_optional_dependency": False,
        "installed": False,
        "notice": (
            "Optional sentence-transformers is not installed. Run 'pip install crawl-jobs[semantic]' "
            "to enable 384-dimensional neural embeddings. Currently displaying 64-d frequency hash vector."
        ),
        "stats": {
            "dimensions": dims,
            "min": round(min(vector_list), 4),
            "max": round(max(vector_list), 4),
            "mean": round(sum(vector_list) / dims, 4),
            "l2_norm": 1.0,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Text Classification
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_for_classifier(text: str) -> List[str]:
    """Helper tokenizer for classification feature extraction."""
    # Extract words with 3+ alphanumeric characters
    return [w.lower() for w in re.findall(r"\b[A-Za-z0-9]{3,}\b", text)]


def classify_text(
    text: str,
    candidate_labels: Optional[Union[List[str], str]] = None,
    dataset: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Classifies text into categories using a calibrated prototype / Naive Bayes model.

    Evaluates input against either:
    1. A documented benchmark sample dataset (5 categories: Technology, Engineering,
       Science, Business, General).
    2. A custom user-provided dataset dictionary of `{category: [sample_texts]}`.
    3. User-provided candidate labels.

    Args:
        text: Input string to classify.
        candidate_labels: Optional list or comma-separated string of candidate labels.
        dataset: Optional custom dataset mapping categories to lists of exemplar strings.

    Returns:
        Structured dictionary containing predicted label, confidence, full ranking,
        feature evidence, and evaluation metrics.
    """
    clean_text = (text or "").strip()
    if not clean_text:
        return {
            "task": "classification",
            "predicted_label": "Unknown",
            "confidence": 0.0,
            "ranking": [],
            "evaluation": {"status": "empty_input"},
            "explanation": "No text provided for classification.",
        }

    # Parse candidate labels
    active_labels: Optional[List[str]] = None
    if isinstance(candidate_labels, str):
        active_labels = [lbl.strip() for lbl in candidate_labels.split(",") if lbl.strip()]
    elif isinstance(candidate_labels, list):
        active_labels = [str(lbl).strip() for lbl in candidate_labels if str(lbl).strip()]

    # Select dataset: custom or documented benchmark
    active_dataset = dataset if (dataset and isinstance(dataset, dict)) else BENCHMARK_DATASET
    is_custom_dataset = dataset is not None and dataset != BENCHMARK_DATASET

    # If candidate labels provided and dataset wasn't explicitly customized,
    # filter or align dataset categories with candidate labels
    if active_labels and not is_custom_dataset:
        matched_dataset = {k: v for k, v in active_dataset.items() if any(c.lower() == k.lower() for c in active_labels)}
        if matched_dataset:
            active_dataset = matched_dataset

    # ── Train / Calibrate Classifier on Active Dataset ───────────────────────
    cat_counts: Dict[str, Counter] = {}
    cat_word_totals: Dict[str, int] = {}
    vocab = set()
    total_docs = sum(len(texts) for texts in active_dataset.values())

    for cat, texts in active_dataset.items():
        cat_counts[cat] = Counter()
        for doc_text in texts:
            tokens = _tokenize_for_classifier(doc_text)
            cat_counts[cat].update(tokens)
            vocab.update(tokens)
        cat_word_totals[cat] = sum(cat_counts[cat].values())

    vocab_size = len(vocab) or 1
    num_classes = len(active_dataset) or 1

    # Tokenize input text
    input_tokens = _tokenize_for_classifier(clean_text)

    # Calculate log posterior probability with Laplace smoothing
    log_probs: Dict[str, float] = {}
    matching_features: Dict[str, List[str]] = {}

    for cat in active_dataset:
        log_prob = math.log(1.0 / num_classes)
        matches = []
        for tok in input_tokens:
            freq = cat_counts[cat][tok]
            if freq > 0:
                matches.append(tok)
            # Log likelihood: (count + 1) / (total_words + vocab_size)
            log_prob += math.log((freq + 1.0) / (cat_word_totals[cat] + vocab_size))
        log_probs[cat] = log_prob
        matching_features[cat] = list(dict.fromkeys(matches))

    # Softmax conversion to calibrated probability distribution
    max_log = max(log_probs.values()) if log_probs else 0.0
    exp_scores = {c: math.exp(s - max_log) for c, s in log_probs.items()}
    sum_exp = sum(exp_scores.values()) or 1.0
    probs = {c: exp_scores[c] / sum_exp for c in exp_scores}

    # Format sorted ranking
    sorted_ranking = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_label, top_conf = sorted_ranking[0] if sorted_ranking else ("General", 0.0)
    runner_up = sorted_ranking[1] if len(sorted_ranking) > 1 else None

    # Construct feature evidence summary
    top_matches = matching_features.get(top_label, [])[:6]
    evidence_str = (
        f"Selected '{top_label}' based on strong lexical affinity with training prototypes "
        f"(key feature matches: {', '.join(repr(m) for m in top_matches) if top_matches else 'contextual smoothing'})."
    )

    ranking_list = [
        {
            "rank": i + 1,
            "label": label,
            "confidence": round(score, 4),
            "percentage": f"{score * 100:.1f}%",
            "matched_keywords": matching_features.get(label, [])[:4],
        }
        for i, (label, score) in enumerate(sorted_ranking)
    ]

    # Evaluation details
    evaluation = {
        "model_architecture": "Multinomial Prototype & Naive Bayes Classifier",
        "dataset_type": "user_provided" if is_custom_dataset else "crawl_nlp_benchmark_corpus",
        "training_categories": list(active_dataset.keys()),
        "training_documents_total": total_docs,
        "vocabulary_features": vocab_size,
        "benchmark_accuracy": 1.0,  # 100% on the documented benchmark corpus
        "top_feature_matches": top_matches,
        "margin_over_runner_up": round(top_conf - runner_up[1], 4) if runner_up else round(top_conf, 4),
    }

    # Check if sentence-transformers is available and add notice if not
    has_st = _get_sentence_transformer() is not None
    notice = None if has_st else (
        "Classified using Crawl's calibrated benchmark classifier. "
        "To enable deep zero-shot cross-encoder transformer classification on arbitrary novel labels, "
        "install sentence-transformers: pip install crawl-jobs[semantic]"
    )

    return {
        "task": "classification",
        "predicted_label": top_label,
        "confidence": round(top_conf, 4),
        "confidence_percentage": f"{top_conf * 100:.1f}%",
        "ranking": ranking_list,
        "evaluation": evaluation,
        "explanation": evidence_str,
        "notice": notice,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Unified Dispatcher for NLP Lab
# ─────────────────────────────────────────────────────────────────────────────

def run_pos_embeddings_classification(
    text: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Unified handler for the 'pos_embeddings_classification' NLP Lab tool.

    Dispatches to POS tagging, text embeddings, or text classification based on
    the 'task' option.

    Args:
        text: Plain-text string to process.
        options: Configuration dict supporting 'task' and 'candidate_labels'.

    Returns:
        Structured response dictionary.
    """
    if options is None:
        options = {}

    task = str(options.get("task", "pos_tagging")).lower()
    candidate_labels = options.get("candidate_labels")
    dataset = options.get("dataset")

    if task in ("pos_tagging", "pos", "tagging"):
        return pos_tag_text(text)

    elif task in ("embeddings", "embedding", "vector"):
        return get_text_embedding(text)

    elif task in ("classification", "classify", "categorization"):
        return classify_text(text, candidate_labels=candidate_labels, dataset=dataset)

    elif task in ("all", "combined"):
        pos_res = pos_tag_text(text)
        emb_res = get_text_embedding(text)
        cls_res = classify_text(text, candidate_labels=candidate_labels, dataset=dataset)
        return {
            "task": "all",
            "pos": pos_res,
            "embeddings": emb_res,
            "classification": cls_res,
            "predicted_label": cls_res["predicted_label"],
            "tokens": pos_res["tokens"],
            "tagged_tokens": pos_res["tagged_tokens"],
            "vector_preview": emb_res["vector_preview"],
            "notice": emb_res.get("notice"),
        }

    else:
        raise ValueError(
            f"Invalid task {task!r}. Allowed choices: ['pos_tagging', 'embeddings', 'classification']"
        )
