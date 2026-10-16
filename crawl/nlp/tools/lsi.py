"""Latent Semantic Indexing (LSI) Analysis Tool.

Applies Latent Semantic Indexing (LSI / LSA) using a TF-IDF document-term matrix
followed by Truncated Singular Value Decomposition (SVD) on document collections.

Key capabilities:
1. Document-Term TF-IDF Matrix Construction.
2. Truncated SVD (X = U * Sigma * V^T) for latent semantic dimension reduction.
3. Semantic Query Retrieval: projects search queries into latent space and ranks
   documents by cosine similarity.
4. Latent Topic / Component Extraction: extracts dominant vocabulary terms for each topic.
5. Document Categorization / Clustering: assigns documents to their primary latent topic.
6. Robust validation of collection size and component count (k) to ensure small inputs
   never crash.
7. Clear architectural distinction from sentence-transformer neural embeddings.
"""

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Try importing numpy for optimized SVD; provide pure-Python SVD fallback
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

# ── Standard English Stopwords ────────────────────────────────────────────────
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


def _split_into_documents(text: str) -> List[str]:
    """Splits text into separate documents via double newlines, numbered prefixes, or lines."""
    raw = (text or "").strip()
    if not raw:
        return []

    # 1. Double newlines (paragraphs)
    blocks = re.split(r"\n\s*\n+", raw)
    cleaned = [b.strip() for b in blocks if b.strip()]
    if len(cleaned) > 1:
        return cleaned

    # 2. Check for explicit "Document X:" prefixes
    if re.search(r"(?im)^Document\s+\d+[:\.]", raw):
        parts = re.split(r"(?im)^Document\s+\d+[:\.]\s*", raw)
        doc_parts = [p.strip() for p in parts if p.strip()]
        if len(doc_parts) >= 2:
            return doc_parts

    # 3. Check for single newline separated lines
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) >= 2:
        return lines

    return [raw]


def _tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Extracts lowercase tokens, optionally filtering stopwords."""
    pattern = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9_+#/'-]*\b")
    tokens = [w.lower() for w in pattern.findall(text)]
    if remove_stopwords:
        tokens = [w for w in tokens if w not in _STOPWORDS and len(w) > 1]
    return tokens


def _pure_python_truncated_svd(
    A: List[List[float]],
    k: int,
    max_iter: int = 60,
    tol: float = 1e-7,
) -> Tuple[List[List[float]], List[float], List[List[float]]]:
    """Pure-Python power-iteration Truncated SVD for fallback when numpy is unavailable.

    Computes top k singular triples for matrix A of size N x M.
    Returns:
        U: N x k left singular vectors (columns)
        S: k singular values
        Vt: k x M right singular vectors (rows)
    """
    N = len(A)
    M = len(A[0]) if N > 0 else 0
    k_eff = max(1, min(k, N, M))

    A_work = [row[:] for row in A]
    u_cols: List[List[float]] = []
    s_vals: List[float] = []
    vt_rows: List[List[float]] = []

    for comp in range(k_eff):
        # Initial unit vector
        v = [1.0 / math.sqrt(M)] * M if M > 0 else []
        u = [0.0] * N
        sigma = 0.0

        for _ in range(max_iter):
            # u = A * v
            u_next = [sum(A_work[i][j] * v[j] for j in range(M)) for i in range(N)]
            norm_u = math.sqrt(sum(x * x for x in u_next))
            if norm_u < 1e-12:
                break
            u = [x / norm_u for x in u_next]

            # v = A^T * u
            v_next = [sum(A_work[i][j] * u[i] for i in range(N)) for j in range(M)]
            norm_v = math.sqrt(sum(x * x for x in v_next))
            if norm_v < 1e-12:
                break
            v = [x / norm_v for x in v_next]
            sigma = norm_v

        u_cols.append(u)
        s_vals.append(sigma)
        vt_rows.append(v)

        # Deflate A_work: A_work = A_work - sigma * u * v^T
        for i in range(N):
            for j in range(M):
                A_work[i][j] -= sigma * u[i] * v[j]

    # Convert u_cols (list of k columns of length N) to N x k matrix
    U_matrix = [[u_cols[c][r] for c in range(len(u_cols))] for r in range(N)]
    return U_matrix, s_vals, vt_rows


class LatentSemanticIndexing:
    """Latent Semantic Indexing (Truncated SVD on TF-IDF document-term matrices).

    Builds a vector space representation of documents, projects them onto
    a lower-dimensional latent topic space via SVD, and enables semantic
    query search and topic categorization.
    """

    def __init__(
        self,
        num_topics: int = 2,
        remove_stopwords: bool = True,
    ) -> None:
        self.num_topics = max(1, int(num_topics))
        self.remove_stopwords = remove_stopwords

        # Model state after fit
        self.is_fitted: bool = False
        self.documents: List[str] = []
        self.vocabulary: List[str] = []
        self.term_to_idx: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

        # SVD Components
        self.k_effective: int = 1
        self.k_adjustment_note: Optional[str] = None
        self.singular_values: List[float] = []
        self.variance_explained_ratio: List[float] = []
        self.doc_coordinates: List[List[float]] = []  # N x k
        self.V_t: List[List[float]] = []  # k x M
        self.engine: str = "numpy.linalg.svd" if _HAS_NUMPY else "pure_python_truncated_svd"

    def fit(self, documents: Union[str, List[str]]) -> "LatentSemanticIndexing":
        """Builds TF-IDF document-term matrix and applies Truncated SVD.

        Args:
            documents: Raw string containing multiple documents, or a list of strings.

        Returns:
            Self instance.
        """
        if isinstance(documents, str):
            doc_list = _split_into_documents(documents)
        else:
            doc_list = [d.strip() for d in documents if d and d.strip()]

        self.documents = doc_list
        N = len(self.documents)

        # Tokenize documents
        tokenized_docs = [_tokenize(doc, self.remove_stopwords) for doc in self.documents]

        # Build vocabulary & document frequency
        df_counter: Dict[str, int] = {}
        for tokens in tokenized_docs:
            for term in set(tokens):
                df_counter[term] = df_counter.get(term, 0) + 1

        self.vocabulary = sorted(df_counter.keys())
        self.term_to_idx = {term: idx for idx, term in enumerate(self.vocabulary)}
        M = len(self.vocabulary)

        # Validate collection size and vocabulary
        if N == 0 or M == 0:
            self.is_fitted = True
            self.k_effective = 0
            self.singular_values = []
            self.doc_coordinates = []
            self.V_t = []
            return self

        # Compute IDF with smooth formulation: log((N + 1) / (DF + 1)) + 1
        self.idf = {
            term: math.log((N + 1.0) / (df_counter[term] + 1.0)) + 1.0
            for term in self.vocabulary
        }

        # Build TF-IDF matrix (N x M)
        tf_idf_matrix: List[List[float]] = []
        for tokens in tokenized_docs:
            doc_counts: Dict[str, int] = {}
            for t in tokens:
                doc_counts[t] = doc_counts.get(t, 0) + 1

            row = [0.0] * M
            for term, count in doc_counts.items():
                if term in self.term_to_idx:
                    col = self.term_to_idx[term]
                    row[col] = float(count) * self.idf[term]

            # Unit normalize document row
            norm = math.sqrt(sum(x * x for x in row)) or 1.0
            tf_idf_matrix.append([x / norm for x in row])

        # Validate and bound k (cannot exceed min(N, M))
        max_possible_rank = min(N, M)
        requested_k = self.num_topics
        if requested_k > max_possible_rank:
            self.k_effective = max_possible_rank
            self.k_adjustment_note = (
                f"Requested component count k={requested_k} was adjusted to {self.k_effective} "
                f"(maximum possible rank for {N} document{'s' if N != 1 else ''} "
                f"and {M} vocabulary term{'s' if M != 1 else ''})."
            )
        else:
            self.k_effective = max(1, requested_k)
            self.k_adjustment_note = None

        k = self.k_effective

        # Perform Truncated SVD
        if _HAS_NUMPY:
            try:
                X_np = np.array(tf_idf_matrix, dtype=float)
                U, S, Vt = np.linalg.svd(X_np, full_matrices=False)

                # Truncate to k
                Uk = U[:, :k]
                Sk = S[:k]
                Vtk = Vt[:k, :]

                # Document coordinates in latent space: D = Uk * Sk (shape N x k)
                Dk = Uk * Sk

                self.singular_values = [round(float(s), 4) for s in Sk]
                self.doc_coordinates = [
                    [round(float(val), 4) for val in row] for row in Dk
                ]
                self.V_t = [[float(val) for val in row] for row in Vtk]
                self.engine = "numpy.linalg.svd"

            except Exception:  # pragma: no cover
                # Fallback to pure-Python SVD if numpy fails
                U_pure, S_pure, Vt_pure = _pure_python_truncated_svd(tf_idf_matrix, k)
                self.singular_values = [round(float(s), 4) for s in S_pure]
                self.doc_coordinates = [
                    [round(U_pure[i][c] * S_pure[c], 4) for c in range(len(S_pure))]
                    for i in range(N)
                ]
                self.V_t = Vt_pure
                self.engine = "pure_python_truncated_svd"
        else:  # pragma: no cover
            U_pure, S_pure, Vt_pure = _pure_python_truncated_svd(tf_idf_matrix, k)
            self.singular_values = [round(float(s), 4) for s in S_pure]
            self.doc_coordinates = [
                [round(U_pure[i][c] * S_pure[c], 4) for c in range(len(S_pure))]
                for i in range(N)
            ]
            self.V_t = Vt_pure
            self.engine = "pure_python_truncated_svd"

        # Variance explained ratio: s_i^2 / sum(s_j^2)
        total_sq = sum(s * s for s in self.singular_values) or 1.0
        self.variance_explained_ratio = [
            round((s * s) / total_sq, 4) for s in self.singular_values
        ]

        self.is_fitted = True
        return self

    def get_topics(self, num_terms: int = 5) -> List[Dict[str, Any]]:
        """Extracts top representative vocabulary terms for each latent topic component.

        Args:
            num_terms: Maximum number of salient terms to include per topic.

        Returns:
            List of topic descriptors with singular values and term loadings.
        """
        if not self.is_fitted or not self.V_t or not self.vocabulary:
            return []

        topics: List[Dict[str, Any]] = []
        for c in range(self.k_effective):
            v_row = self.V_t[c]
            # Rank terms by absolute weight loading
            indexed_weights = [(self.vocabulary[j], v_row[j]) for j in range(len(self.vocabulary))]
            sorted_terms = sorted(indexed_weights, key=lambda x: abs(x[1]), reverse=True)

            top_terms_list = [
                {"term": term, "weight": round(weight, 4)}
                for term, weight in sorted_terms[:num_terms]
            ]
            s_val = self.singular_values[c] if c < len(self.singular_values) else 0.0
            var_pct = (
                f"{self.variance_explained_ratio[c] * 100:.1f}%"
                if c < len(self.variance_explained_ratio)
                else "0.0%"
            )

            topics.append({
                "topic_id": f"Topic {c + 1}",
                "topic_index": c + 1,
                "singular_value": s_val,
                "variance_explained": var_pct,
                "top_terms": top_terms_list,
                "summary": ", ".join(f"{t['term']} ({t['weight']:+.3f})" for t in top_terms_list[:4]),
            })

        return topics

    def get_document_coordinates(self) -> List[Dict[str, Any]]:
        """Returns the coordinates of all documents in the reduced latent space.

        Returns:
            List of document descriptors with latent vectors and dominant topic.
        """
        if not self.is_fitted or not self.doc_coordinates:
            return []

        docs_info: List[Dict[str, Any]] = []
        for i, coords in enumerate(self.doc_coordinates):
            snippet = self.documents[i][:90] + ("..." if len(self.documents[i]) > 90 else "")

            # Dominant topic is topic with largest absolute loading
            if coords:
                dominant_idx = max(range(len(coords)), key=lambda c: abs(coords[c]))
                dominant_topic = f"Topic {dominant_idx + 1}"
            else:
                dominant_topic = "Topic 1"

            docs_info.append({
                "doc_id": f"Doc {i + 1}",
                "index": i,
                "snippet": snippet,
                "coordinates": coords,
                "dominant_topic": dominant_topic,
            })

        return docs_info

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Ranks documents by cosine similarity against the query in latent semantic space.

        Args:
            query: User search query string.
            top_k: Limit number of ranked documents returned (default: all).

        Returns:
            Ranked list of documents with similarity scores, percentage, and coordinates.
        """
        if not self.is_fitted or not self.doc_coordinates or not self.vocabulary:
            return []

        # Tokenize query
        q_tokens = _tokenize(query, self.remove_stopwords)
        M = len(self.vocabulary)
        q_vec = [0.0] * M

        matched_query_terms: List[str] = []
        for t in q_tokens:
            if t in self.term_to_idx:
                col = self.term_to_idx[t]
                q_vec[col] += 1.0 * self.idf.get(t, 1.0)
                matched_query_terms.append(t)

        matched_query_terms = list(dict.fromkeys(matched_query_terms))
        q_norm = math.sqrt(sum(x * x for x in q_vec))

        # Project query into latent space: q_k = q * V_t^T (size k)
        k = self.k_effective
        if q_norm < 1e-12 or not self.V_t:
            # Query has no vocabulary overlap; similarity is 0.0
            q_latent = [0.0] * k
            is_oov = True
        else:
            q_unit = [x / q_norm for x in q_vec]
            q_latent = [
                sum(q_unit[j] * self.V_t[c][j] for j in range(M))
                for c in range(k)
            ]
            is_oov = False

        q_latent_norm = math.sqrt(sum(x * x for x in q_latent))

        # Calculate cosine similarity with each document's latent vector
        results: List[Dict[str, Any]] = []
        for i, d_coords in enumerate(self.doc_coordinates):
            snippet = self.documents[i][:90] + ("..." if len(self.documents[i]) > 90 else "")

            if is_oov or q_latent_norm < 1e-12:
                sim = 0.0
            else:
                d_norm = math.sqrt(sum(x * x for x in d_coords))
                if d_norm < 1e-12:
                    sim = 0.0
                else:
                    dot = sum(d_coords[c] * q_latent[c] for c in range(k))
                    sim = dot / (d_norm * q_latent_norm)
                    sim = max(-1.0, min(1.0, sim))

            # Scaled positive similarity percentage for intuitive display
            pct = f"{max(0.0, sim) * 100:.1f}%"

            results.append({
                "doc_id": f"Doc {i + 1}",
                "index": i,
                "snippet": snippet,
                "similarity": round(sim, 4),
                "similarity_percentage": pct,
                "coordinates": d_coords,
                "dominant_topic": (
                    f"Topic {max(range(len(d_coords)), key=lambda c: abs(d_coords[c])) + 1}"
                    if d_coords else "Topic 1"
                ),
            })

        # Rank descending by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        for rank, res in enumerate(results, 1):
            res["rank"] = rank

        return results[:top_k] if top_k else results

    def categorize(self) -> List[Dict[str, Any]]:
        """Groups documents into semantic clusters based on their dominant latent topic.

        Returns:
            List of topic clusters with assigned documents and key terms.
        """
        topics = self.get_topics()
        docs = self.get_document_coordinates()

        clusters: Dict[str, List[Dict[str, Any]]] = {
            t["topic_id"]: [] for t in topics
        }

        for d in docs:
            topic_key = d["dominant_topic"]
            if topic_key not in clusters:
                clusters[topic_key] = []
            clusters[topic_key].append(d)

        categorized_results: List[Dict[str, Any]] = []
        for t in topics:
            assigned = clusters.get(t["topic_id"], [])
            categorized_results.append({
                "topic_id": t["topic_id"],
                "variance_explained": t["variance_explained"],
                "singular_value": t["singular_value"],
                "top_terms": t["top_terms"],
                "summary": t["summary"],
                "assigned_documents_count": len(assigned),
                "assigned_documents": assigned,
            })

        return categorized_results


# ── Unified Public Functional API ─────────────────────────────────────────────

def run_lsi_analysis(
    documents: Union[str, List[str]],
    query: str = "scalable backend microservices",
    num_topics: int = 2,
    mode: str = "search",
    show_coordinates: bool = True,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Runs Latent Semantic Indexing (Truncated SVD on TF-IDF document-term matrix).

    Args:
        documents: Raw text block or list of document strings.
        query: Search query string for similarity ranking (search mode).
        num_topics: Target number of SVD latent components (default: 2).
        mode: 'search' (default) or 'categorize'.
        show_coordinates: If True, include document coordinate vectors in latent space.
        options: Additional parameters dict.

    Returns:
        Structured result dictionary with topics, coordinates, rankings, and explanations.
    """
    if options is None:
        options = {}

    opt_mode = options.get("mode", mode)
    opt_query = options.get("query", query)
    opt_num_topics = int(options.get("num_topics", num_topics))
    opt_show_coords = options.get("show_coordinates", show_coordinates)

    # Initialize model
    model = LatentSemanticIndexing(num_topics=opt_num_topics, remove_stopwords=True)
    model.fit(documents)

    N = len(model.documents)
    M = len(model.vocabulary)

    # Handle boundary case: empty collection or empty vocabulary
    if N == 0:
        return {
            "task": "lsi_search",
            "mode": opt_mode,
            "query": opt_query,
            "num_documents": 0,
            "vocabulary_size": 0,
            "num_topics_requested": opt_num_topics,
            "num_topics_used": 0,
            "k_adjustment_note": "No documents provided.",
            "engine": model.engine,
            "topics": [],
            "documents": [],
            "rankings": [],
            "categories": [],
            "explanation": "No documents provided. Please input at least one document (double newline separated).",
            "lsi_label_notice": (
                "Latent Semantic Indexing (LSI) uses linear algebraic Truncated SVD on TF-IDF term-document matrices "
                "to discover hidden concept correlations, distinct from neural sentence-transformers."
            ),
        }

    if M == 0:
        return {
            "task": "lsi_search",
            "mode": opt_mode,
            "query": opt_query,
            "num_documents": N,
            "vocabulary_size": 0,
            "num_topics_requested": opt_num_topics,
            "num_topics_used": 0,
            "k_adjustment_note": "No vocabulary remaining after filtering.",
            "engine": model.engine,
            "topics": [],
            "documents": [],
            "rankings": [],
            "categories": [],
            "explanation": "No terms found after stopword filtering. Try inputting longer text with lexical keywords.",
            "lsi_label_notice": (
                "Latent Semantic Indexing (LSI) uses linear algebraic Truncated SVD on TF-IDF term-document matrices "
                "to discover hidden concept correlations, distinct from neural sentence-transformers."
            ),
        }

    topics = model.get_topics()
    doc_coords = model.get_document_coordinates() if opt_show_coords else []

    # Execute specified mode
    if opt_mode == "categorize":
        categories = model.categorize()
        rankings = []
        explanation = (
            f"Clustered {N} documents across {model.k_effective} latent SVD topic components. "
            f"Each document is categorized by its highest coordinate projection in the reduced semantic space."
        )
    else:  # 'search' mode
        rankings = model.search(opt_query)
        categories = []
        top_match = rankings[0] if rankings else None
        top_doc_label = f"{top_match['doc_id']} ({top_match['similarity_percentage']})" if top_match else "None"
        explanation = (
            f"Ranked {N} documents against query '{opt_query}' using Truncated SVD (k={model.k_effective}). "
            f"Top match: {top_doc_label}. Queries and documents are mapped into the same latent concept space, "
            f"overcoming synonymy and lexical mismatch."
        )

    return {
        "task": "lsi_search",
        "mode": opt_mode,
        "query": opt_query,
        "num_documents": N,
        "vocabulary_size": M,
        "num_topics_requested": opt_num_topics,
        "num_topics_used": model.k_effective,
        "k_adjustment_note": model.k_adjustment_note,
        "engine": model.engine,
        "singular_values": model.singular_values,
        "variance_explained": model.variance_explained_ratio,
        "topics": topics,
        "documents": doc_coords,
        "rankings": rankings,
        "categories": categories,
        "explanation": explanation,
        "lsi_label_notice": (
            "Latent Semantic Indexing (LSI) uses linear algebraic Truncated SVD on TF-IDF term-document matrices "
            "to discover hidden concept correlations, distinct from neural sentence-transformers."
        ),
    }


def run_lsi_tool(text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Unified handler for the 'lsi_search' tool in Crawl NLP Lab."""
    return run_lsi_analysis(documents=text, options=options)
