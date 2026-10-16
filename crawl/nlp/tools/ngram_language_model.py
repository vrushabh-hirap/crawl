"""N-gram Language Model & Sequence Probability Estimation.

Features:
1. Builds unigram, bigram, and trigram statistical language models from a supplied text corpus.
2. Calculates exact sequence joint probabilities, log-probabilities, and perplexity.
3. Implements Laplace (Add-1) and Add-k smoothing to handle unseen n-grams and avoid zero probabilities.
4. Generates step-by-step mathematical explanations showing n-gram counts, context counts,
   and smoothing formulas for assignment demonstrations and pedagogical inspection.

Completely self-contained in pure Python with zero heavy dependencies.
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple, Union


def _tokenize_corpus(text: str) -> List[str]:
    """Tokenizes text into lowercase words while preserving apostrophes and hyphens."""
    pattern = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9_+#/'-]*\b")
    return [w.lower() for w in pattern.findall(text or "")]


def _split_into_sentences(text: str) -> List[List[str]]:
    """Splits corpus into sentences, each represented as a list of word tokens."""
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    sentences: List[List[str]] = []
    for s in raw_sentences:
        tokens = _tokenize_corpus(s)
        if tokens:
            sentences.append(tokens)
    return sentences


class NGramLanguageModel:
    """N-gram Statistical Language Model with configurable smoothing.

    Supports orders n=1 (unigram), n=2 (bigram), n=3 (trigram), or higher.
    """

    def __init__(self, n: int = 2, smoothing: str = "laplace", k: float = 1.0) -> None:
        """Initializes the N-gram language model.

        Args:
            n: Model order (1: Unigram, 2: Bigram, 3: Trigram).
            smoothing: Smoothing algorithm ('laplace', 'add_k', or 'none').
            k: Smoothing parameter (k=1.0 for standard Laplace).
        """
        self.n = max(1, int(n))
        self.smoothing = str(smoothing).lower()
        self.k = float(k) if self.smoothing in ("laplace", "add_k") else 0.0
        if self.smoothing == "laplace":
            self.k = 1.0

        # Frequency counters
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.unigram_counts: Counter = Counter()
        self.vocabulary: Set[str] = set()
        self.total_tokens: int = 0
        self.is_trained: bool = False

    @property
    def vocab(self) -> Set[str]:
        """Convenience property alias for self.vocabulary."""
        return self.vocabulary

    def fit(self, corpus: Union[str, List[List[str]]]) -> "NGramLanguageModel":
        """Trains the N-gram language model on the supplied text corpus.

        Args:
            corpus: Raw corpus text string, or a list of tokenized sentences.

        Returns:
            Self instance after fitting.
        """
        if isinstance(corpus, str):
            sentences = _split_into_sentences(corpus)
        else:
            sentences = corpus

        self.ngram_counts.clear()
        self.context_counts.clear()
        self.unigram_counts.clear()
        self.vocabulary.clear()
        self.total_tokens = 0

        pad_token = "<s>"
        end_token = "</s>"

        for sent in sentences:
            if not sent:
                continue

            # Update vocabulary and unigram counts
            for word in sent:
                self.vocabulary.add(word)
                self.unigram_counts[word] += 1
                self.total_tokens += 1

            if self.n == 1:
                # Unigrams
                for word in sent:
                    self.ngram_counts[(word,)] += 1
            else:
                # Pad sentence with n-1 start tokens and 1 end token
                padded = [pad_token] * (self.n - 1) + sent + [end_token]

                # Extract n-grams and (n-1)-gram contexts
                for i in range(len(padded) - self.n + 1):
                    ngram = tuple(padded[i : i + self.n])
                    context = tuple(padded[i : i + self.n - 1])
                    self.ngram_counts[ngram] += 1
                    self.context_counts[context] += 1

        self.is_trained = True
        return self

    def calculate_sequence_probability(self, target_sequence: str) -> Dict[str, Any]:
        """Calculates the probability of a target word sequence under the trained model.

        Args:
            target_sequence: The phrase or sentence to evaluate.

        Returns:
            Dictionary containing sequence probability, log2 probability, perplexity,
            step-by-step count details, and explanatory notes.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained on a corpus using fit() before evaluating sequences.")

        target_tokens = _tokenize_corpus(target_sequence)
        order_name = {1: "Unigram", 2: "Bigram", 3: "Trigram"}.get(self.n, f"{self.n}-gram")
        vocab_size = len(self.vocabulary)

        if not target_tokens:
            return {
                "task": "ngram_probability",
                "sequence": target_sequence,
                "target_sequence": target_sequence,
                "tokens": [],
                "target_tokens": [],
                "tokens_count": 0,
                "target_length": 0,
                "n": self.n,
                "order_name": order_name,
                "smoothing": self.smoothing,
                "smoothing_k": self.k,
                "vocabulary_size": vocab_size,
                "total_corpus_tokens": self.total_tokens,
                "corpus_tokens_count": self.total_tokens,
                "total_probability": 0.0,
                "total_probability_scientific": "0.0",
                "log2_probability": float("-inf"),
                "perplexity": float("inf"),
                "unseen_ngram_count": 0,
                "out_of_vocabulary": [],
                "steps": [],
                "explanation": "Target sequence was empty.",
                "unseen_handling_summary": "Empty sequence provided.",
            }

        pad_token = "<s>"
        steps: List[Dict[str, Any]] = []

        total_log2_prob = 0.0
        collapsed_to_zero = False
        unseen_ngram_count = 0
        oov_words: List[str] = [w for w in target_tokens if w not in self.vocabulary]

        # 1. Unigram Model Evaluation
        if self.n == 1:
            N = self.total_tokens or 1
            for i, word in enumerate(target_tokens):
                c_w = self.unigram_counts[word]
                is_unseen = (c_w == 0)
                if is_unseen:
                    unseen_ngram_count += 1

                if self.smoothing in ("laplace", "add_k"):
                    k_val = self.k
                    prob = (c_w + k_val) / (N + k_val * vocab_size) if (N + k_val * vocab_size) > 0 else 0.0
                    formula = f"({c_w} + {k_val}) / ({N} + {k_val} * {vocab_size})"
                else:
                    prob = (c_w / N) if N > 0 else 0.0
                    formula = f"{c_w} / {N}"

                if prob <= 0.0:
                    collapsed_to_zero = True
                    step_log2 = float("-inf")
                else:
                    step_log2 = math.log2(prob)
                    total_log2_prob += step_log2

                formula_str = f"P({word}) = {formula}"
                steps.append({
                    "step": i + 1,
                    "target_word": word,
                    "context": [],
                    "context_str": "ε",
                    "ngram": word,
                    "ngram_tuple": [word],
                    "ngram_count": c_w,
                    "context_count": N,
                    "vocab_size": vocab_size,
                    "formula": formula,
                    "formula_str": formula_str,
                    "probability": round(prob, 6) if prob > 0 else 0.0,
                    "probability_formatted": f"{prob:.6f}",
                    "log2_prob": round(step_log2, 4) if step_log2 != float("-inf") else "-inf",
                    "log2_probability": round(step_log2, 4) if step_log2 != float("-inf") else "-inf",
                    "status": "seen" if not is_unseen else "unseen",
                    "is_unseen": is_unseen,
                })

        # 2. Bigram, Trigram, or higher N-gram Evaluation
        else:
            padded_target = [pad_token] * (self.n - 1) + target_tokens
            # Evaluate P(w_i | w_{i-n+1}^{i-1})
            for i in range(len(target_tokens)):
                window = padded_target[i : i + self.n]
                ngram = tuple(window)
                context = tuple(window[:-1])
                target_word = window[-1]

                c_ngram = self.ngram_counts[ngram]
                c_context = self.context_counts[context]

                is_unseen = (c_ngram == 0)
                if is_unseen:
                    unseen_ngram_count += 1

                if self.smoothing in ("laplace", "add_k"):
                    k_val = self.k
                    denominator = c_context + k_val * vocab_size
                    prob = (c_ngram + k_val) / denominator if denominator > 0 else 0.0
                    formula = f"({c_ngram} + {k_val}) / ({c_context} + {k_val} * {vocab_size})"
                else:
                    prob = (c_ngram / c_context) if c_context > 0 else 0.0
                    formula = f"{c_ngram} / {c_context}" if c_context > 0 else f"{c_ngram} / 0 (undefined)"

                if prob <= 0.0:
                    collapsed_to_zero = True
                    step_log2 = float("-inf")
                else:
                    step_log2 = math.log2(prob)
                    total_log2_prob += step_log2

                status = "seen"
                if is_unseen:
                    status = "unseen_ngram" if c_context > 0 else "unseen_context"
                if target_word not in self.vocabulary and target_word != pad_token:
                    status += " (OOV)"

                context_str = " ".join(context)
                formula_str = f"P({target_word} | {context_str}) = {formula}"
                steps.append({
                    "step": i + 1,
                    "target_word": target_word,
                    "context": list(context),
                    "context_str": context_str,
                    "ngram": " ".join(ngram),
                    "ngram_tuple": list(ngram),
                    "ngram_count": c_ngram,
                    "context_count": c_context,
                    "vocab_size": vocab_size,
                    "formula": formula,
                    "formula_str": formula_str,
                    "probability": round(prob, 6) if prob > 0 else 0.0,
                    "probability_formatted": f"{prob:.6f}",
                    "log2_prob": round(step_log2, 4) if step_log2 != float("-inf") else "-inf",
                    "log2_probability": round(step_log2, 4) if step_log2 != float("-inf") else "-inf",
                    "status": status,
                    "is_unseen": is_unseen,
                })

        # Calculate final joint probability & perplexity
        m = len(target_tokens)
        if collapsed_to_zero:
            total_prob = 0.0
            log2_prob = float("-inf")
            perplexity = float("inf")
        else:
            log2_prob = round(total_log2_prob, 4)
            # 2^(log2_prob)
            try:
                total_prob = math.pow(2.0, total_log2_prob)
            except OverflowError:
                total_prob = 0.0
            # Perplexity: 2^(-(1/m) * sum(log2 P))
            avg_neg_log = -total_log2_prob / m if m > 0 else 0.0
            perplexity = round(math.pow(2.0, avg_neg_log), 4)

        # Build pedagogical explanation
        if collapsed_to_zero:
            explanation = (
                f"Without smoothing ('none' / MLE), {unseen_ngram_count} unseen {order_name.lower()}s had zero counts. "
                f"By the chain rule of probability, multiplying by zero caused the entire sequence probability to collapse to 0.0 "
                f"(log2 probability = -inf, perplexity = inf). Enable Laplace smoothing to assign valid non-zero probability mass."
            )
        elif unseen_ngram_count > 0:
            explanation = (
                f"Evaluated using {order_name} model with {self.smoothing.capitalize()} smoothing. "
                f"Although {unseen_ngram_count} {order_name.lower()}(s) were never observed in the training corpus (C=0), "
                f"Laplace smoothing assigned baseline probability mass using vocabulary size V={vocab_size}, "
                f"yielding a valid joint probability of {total_prob:.4e} and perplexity of {perplexity}."
            )
        else:
            explanation = (
                f"All {m} {order_name.lower()}s in the target sequence were observed in the training corpus. "
                f"Joint probability: {total_prob:.4e} (log2: {log2_prob}, perplexity: {perplexity})."
            )

        unseen_summary = (
            f"MLE zero-collapse occurred: {unseen_ngram_count} unseen transition(s) caused total probability to collapse to 0.0."
            if collapsed_to_zero
            else (
                f"Laplace smoothing assigned non-zero probabilities to {unseen_ngram_count} unseen transition(s) using vocabulary size V={vocab_size}."
                if unseen_ngram_count > 0
                else "All transitions were observed in the training corpus."
            )
        )

        return {
            "task": "ngram_probability",
            "sequence": target_sequence,
            "target_sequence": target_sequence,
            "tokens": target_tokens,
            "target_tokens": target_tokens,
            "tokens_count": m,
            "target_length": m,
            "n": self.n,
            "order_name": order_name,
            "smoothing": self.smoothing,
            "smoothing_k": self.k,
            "vocabulary_size": vocab_size,
            "total_corpus_tokens": self.total_tokens,
            "corpus_tokens_count": self.total_tokens,
            "total_probability": total_prob,
            "total_probability_scientific": "0.0" if total_prob == 0.0 else f"{total_prob:.6e}",
            "log2_probability": log2_prob,
            "perplexity": perplexity,
            "unseen_ngram_count": unseen_ngram_count,
            "out_of_vocabulary": list(dict.fromkeys(oov_words)),
            "steps": steps,
            "explanation": explanation,
            "unseen_handling_summary": unseen_summary,
        }


# ── Public Functional API ─────────────────────────────────────────────────────

def calculate_sequence_probability(
    corpus: str,
    target_sequence: str,
    n: int = 2,
    smoothing: str = "laplace",
    k: float = 1.0,
) -> Dict[str, Any]:
    """Builds an N-gram model on the corpus and calculates sequence probability.

    Args:
        corpus: The training text to build the model from.
        target_sequence: The test phrase/sentence to evaluate.
        n: Model order (1: Unigram, 2: Bigram, 3: Trigram). Default 2.
        smoothing: 'laplace' (default), 'add_k', or 'none'.
        k: Smoothing factor for 'laplace' (k=1.0) or 'add_k'.

    Returns:
        Structured evaluation dictionary with probability, perplexity, and steps.
    """
    if not corpus or not corpus.strip():
        order_name = {1: "Unigram", 2: "Bigram", 3: "Trigram"}.get(n, f"{n}-gram")
        return {
            "task": "ngram_probability",
            "sequence": target_sequence,
            "target_sequence": target_sequence,
            "tokens": [],
            "target_tokens": [],
            "tokens_count": 0,
            "target_length": 0,
            "n": n,
            "order_name": order_name,
            "smoothing": smoothing,
            "smoothing_k": k,
            "vocabulary_size": 0,
            "total_corpus_tokens": 0,
            "corpus_tokens_count": 0,
            "total_probability": 0.0,
            "total_probability_scientific": "0.0",
            "log2_probability": float("-inf"),
            "perplexity": float("inf"),
            "unseen_ngram_count": 0,
            "out_of_vocabulary": [],
            "steps": [],
            "explanation": "Empty corpus provided. Cannot train an N-gram language model.",
            "unseen_handling_summary": "Empty corpus.",
        }

    model = NGramLanguageModel(n=n, smoothing=smoothing, k=k)
    model.fit(corpus)
    return model.calculate_sequence_probability(target_sequence)


def run_ngram_tool(text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Unified handler for the 'ngram_probability' tool in Crawl NLP Lab.

    Args:
        text: Training corpus input text.
        options: Configuration dictionary supporting 'n', 'smoothing', and 'target_sequence'.

    Returns:
        Structured result dictionary.
    """
    if options is None:
        options = {}

    n = int(options.get("n", 2))
    smoothing = str(options.get("smoothing", "laplace")).lower()
    k = float(options.get("k", 1.0))
    target_seq = options.get("target_sequence")

    corpus = text
    # Support inline separation: "Corpus text... --- TARGET --- target phrase"
    if "--- TARGET ---" in text:
        parts = text.split("--- TARGET ---")
        corpus = parts[0].strip()
        target_seq = parts[1].strip()

    if not target_seq or not str(target_seq).strip():
        # Fallback to evaluating the first sentence or default sample sequence
        sentences = _split_into_sentences(corpus)
        if sentences:
            target_seq = " ".join(sentences[0][:6])
        else:
            target_seq = "to be or not to be"

    result = calculate_sequence_probability(
        corpus=corpus,
        target_sequence=str(target_seq),
        n=n,
        smoothing=smoothing,
        k=k,
    )
    result["tokens"] = result["tokens"]
    return result
