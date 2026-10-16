"""TF-IDF (Term Frequency - Inverse Document Frequency) Analysis Tool.

Computes TF-IDF matrices and term weights across multi-document collections.
Identifies salient vocabulary and distinctive terms with high discriminatory power
to explain why specific words distinguish one document from another.

Completely independent of the ATS taxonomy matcher and sentence-transformers.
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# ── Standard Stopwords ────────────────────────────────────────────────────────
_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
    "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves",
}


def _split_into_documents(text: str, corpus: Optional[str] = None) -> List[str]:
    """Splits input into individual documents based on double newlines or blank lines."""
    docs: List[str] = []

    # 1. Split primary text by double linebreaks or paragraph delimiters
    raw_blocks = re.split(r"\n\s*\n+", (text or "").strip())
    for block in raw_blocks:
        b = block.strip()
        if b:
            docs.append(b)

    # 2. Add extra corpus documents if provided
    if corpus and corpus.strip():
        extra_blocks = re.split(r"\n\s*\n+", corpus.strip())
        for block in extra_blocks:
            b = block.strip()
            if b:
                docs.append(b)

    # If text had no double linebreaks but has multiple lines/sentences,
    # split by single newline if multiple lines exist
    if len(docs) == 1 and "\n" in docs[0]:
        lines = [line.strip() for line in docs[0].splitlines() if line.strip()]
        if len(lines) >= 2:
            docs = lines

    return docs


def _tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenizes text into lowercase words, optionally filtering stopwords."""
    # Match words, compound terms like 'tf-idf', 'c++', 'k8s', 'front-end'
    pattern = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9_+#/'-]*\b")
    tokens = [w.lower() for w in pattern.findall(text)]
    if remove_stopwords:
        tokens = [w for w in tokens if w not in _STOPWORDS and len(w) > 1]
    return tokens


def _generate_ngrams(tokens: List[str], min_n: int, max_n: int) -> List[str]:
    """Generates word n-grams within the given range."""
    ngrams: List[str] = []
    num_tokens = len(tokens)
    for n in range(min_n, max_n + 1):
        for i in range(num_tokens - n + 1):
            ngram = " ".join(tokens[i : i + n])
            ngrams.append(ngram)
    return ngrams


def _parse_ngram_range(ngram_opt: str) -> Tuple[int, int]:
    """Parses ngram range string like '1-1', '1-2', '1-3' into (min_n, max_n)."""
    if not ngram_opt or not isinstance(ngram_opt, str):
        return 1, 1
    parts = ngram_opt.replace(" ", "").split("-")
    try:
        if len(parts) == 2:
            return max(1, int(parts[0])), max(1, int(parts[1]))
        elif len(parts) == 1:
            val = max(1, int(parts[0]))
            return val, val
    except (ValueError, TypeError):
        pass
    return 1, 1


# ── Public TF-IDF Function ────────────────────────────────────────────────────

def compute_tfidf(
    text: Union[str, List[str]],
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Computes TF-IDF matrix, vocabulary weights, and distinguishing terms across documents.

    Args:
        text: A single string with documents separated by double linebreaks, or a list of strings.
        options: Optional dict supporting:
            - 'ngram_range': '1-1' (default), '1-2', or '1-3'
            - 'max_features': max vocabulary terms to keep (default: 20)
            - 'corpus': extra reference documents separated by double linebreaks
            - 'remove_stopwords': bool (default: True)

    Returns:
        Structured dictionary containing:
            - num_documents: int count of processed documents
            - vocabulary_size: total distinct terms across the corpus
            - vocabulary: list of vocabulary strings
            - documents: per-document breakdowns with top terms and distinctive scores
            - matrix: top vocabulary terms with IDF and per-document TF-IDF weights
            - distinguishing_analysis: list of human-readable insights explaining
              which terms best differentiate each document
    """
    if options is None:
        options = {}

    ngram_str = str(options.get("ngram_range", "1-1"))
    min_n, max_n = _parse_ngram_range(ngram_str)
    max_features = int(options.get("max_features", 20))
    remove_stopwords = bool(options.get("remove_stopwords", True))
    corpus_extra = options.get("corpus")

    # 1. Normalize into list of documents
    if isinstance(text, list):
        documents = [str(d).strip() for d in text if str(d).strip()]
        if corpus_extra and isinstance(corpus_extra, str):
            extra = _split_into_documents(corpus_extra)
            documents.extend(extra)
    else:
        documents = _split_into_documents(text, corpus=corpus_extra)

    num_docs = len(documents)
    if num_docs == 0:
        return {
            "num_documents": 0,
            "vocabulary_size": 0,
            "vocabulary": [],
            "documents": [],
            "matrix": [],
            "distinguishing_analysis": ["No document text provided."],
            "tokens": [],
        }

    # 2. Tokenize and extract n-grams per document
    doc_ngrams: List[List[str]] = []
    doc_counts: List[Counter] = []
    global_df = Counter()

    for doc in documents:
        tokens = _tokenize(doc, remove_stopwords=remove_stopwords)
        ngrams = _generate_ngrams(tokens, min_n=min_n, max_n=max_n)
        counts = Counter(ngrams)
        doc_ngrams.append(ngrams)
        doc_counts.append(counts)
        # Document frequency: number of documents where term appears
        global_df.update(set(ngrams))

    all_terms = list(global_df.keys())
    vocab_size = len(all_terms)

    # 3. Calculate Smooth IDF: ln((1 + N) / (1 + DF)) + 1
    # This prevents division by zero and ensures common terms don't get 0 weight
    idf: Dict[str, float] = {}
    for term, df_val in global_df.items():
        idf[term] = math.log((1.0 + num_docs) / (1.0 + df_val)) + 1.0

    # 4. Calculate TF-IDF vectors per document with L2 normalization
    # w(t, d) = tf(t, d) * idf(t)
    # w_norm(t, d) = w(t, d) / sqrt(sum(w^2))
    doc_tfidf: List[Dict[str, float]] = []
    for counts in doc_counts:
        raw_vec: Dict[str, float] = {}
        for term, count in counts.items():
            raw_vec[term] = float(count) * idf[term]

        norm = math.sqrt(sum(v * v for v in raw_vec.values())) or 1.0
        norm_vec = {term: round(v / norm, 4) for term, v in raw_vec.items()}
        doc_tfidf.append(norm_vec)

    # 5. Determine Top Features across the whole corpus
    # Rank terms by their aggregate TF-IDF weight across documents
    aggregate_scores = Counter()
    for d_vec in doc_tfidf:
        aggregate_scores.update(d_vec)

    sorted_features = [term for term, _ in aggregate_scores.most_common(max_features)]

    # 6. Build Matrix for Table Display
    matrix_rows = []
    for term in sorted_features:
        scores_for_docs = [doc_tfidf[i].get(term, 0.0) for i in range(num_docs)]
        matrix_rows.append({
            "term": term,
            "df": global_df[term],
            "idf": round(idf[term], 4),
            "aggregate_score": round(aggregate_scores[term], 4),
            "doc_scores": scores_for_docs,
        })

    # 7. Analyze Distinguishing Terms per Document
    # A distinguishing term has high TF-IDF in this document and low or zero in others
    doc_summaries = []
    distinguishing_insights = []

    for i, doc in enumerate(documents):
        doc_label = f"Doc {i + 1}"
        vec = doc_tfidf[i]
        sorted_doc_terms = sorted(vec.items(), key=lambda x: x[1], reverse=True)

        top_terms_list = [
            {
                "term": t,
                "score": score,
                "tf": doc_counts[i][t],
                "idf": round(idf[t], 4),
            }
            for t, score in sorted_doc_terms[:8]
        ]

        # Calculate distinctiveness: score in doc i minus max score in other docs
        distinctive_candidates = []
        for t, score in sorted_doc_terms:
            max_other = max((doc_tfidf[j].get(t, 0.0) for j in range(num_docs) if j != i), default=0.0)
            diff = score - max_other
            distinctive_candidates.append((t, score, diff, max_other))

        distinctive_candidates.sort(key=lambda x: x[2], reverse=True)
        top_distinctive = [
            {"term": t, "score": score, "uniqueness_margin": round(diff, 4)}
            for t, score, diff, _ in distinctive_candidates[:5]
            if diff > 0.05 or num_docs == 1
        ]

        snippet = doc[:90] + ("..." if len(doc) > 90 else "")
        doc_summaries.append({
            "doc_id": doc_label,
            "snippet": snippet,
            "tokens_count": len(doc_ngrams[i]),
            "top_terms": top_terms_list,
            "distinctive_terms": top_distinctive,
        })

        # Add human-readable insight
        if top_distinctive:
            lead_term = top_distinctive[0]["term"]
            lead_score = top_distinctive[0]["score"]
            margin = top_distinctive[0]["uniqueness_margin"]
            df_val = global_df[lead_term]
            distinguishing_insights.append(
                f"{doc_label}: '{lead_term}' uniquely distinguishes this document with TF-IDF weight {lead_score} "
                f"(margin +{margin} over other documents, appearing in {df_val}/{num_docs} documents)."
            )
        elif sorted_doc_terms:
            lead_term = sorted_doc_terms[0][0]
            distinguishing_insights.append(
                f"{doc_label}: Top weighted term is '{lead_term}' (TF-IDF: {sorted_doc_terms[0][1]})."
            )

    return {
        "num_documents": num_docs,
        "vocabulary_size": vocab_size,
        "vocabulary": sorted_features,
        "documents": doc_summaries,
        "matrix": matrix_rows,
        "distinguishing_analysis": distinguishing_insights,
        "tokens": sorted_features,  # For UI tokens pill box!
        "options_used": {
            "ngram_range": ngram_str,
            "max_features": max_features,
            "remove_stopwords": remove_stopwords,
        },
    }
