"""NLP Lab: Interactive Natural Language Processing and Text Analytics.

Provides registry, schemas, validation, and execution hooks for NLP tools:
- Text analysis (language detection, word/sentence count, tokens)
- Stemming and lemmatization (WordNet, spaCy, Porter)
- POS tagging, embeddings, and text classification
- TF-IDF analysis
- N-gram sequence probability
- LSI document search and categorization
"""

from crawl.nlp.schemas import NLPTool, ToolOption, NLPRequest, NLPResponse
from crawl.nlp.registry import get_tool_registry, get_tool_by_id
from crawl.nlp.service import (
    validate_nlp_request,
    execute_nlp_tool,
    register_tool_handler,
    unregister_tool_handler,
    is_tool_connected,
)
from crawl.nlp.tools.text_analysis import (
    analyze_text,
    detect_language,
    tokenize_text,
    split_sentences,
    count_words,
    count_sentences,
)
from crawl.nlp.tools.stem_lemma import stem_and_lemmatize
from crawl.nlp.tools.pos_embeddings_classification import (
    pos_tag_text,
    get_text_embedding,
    classify_text,
    run_pos_embeddings_classification,
)
from crawl.nlp.tools.tfidf import compute_tfidf
from crawl.nlp.tools.ngram_language_model import (
    NGramLanguageModel,
    calculate_sequence_probability,
    run_ngram_tool,
)
from crawl.nlp.tools.lsi import (
    LatentSemanticIndexing,
    run_lsi_analysis,
    run_lsi_tool,
)

__all__ = [
    "NLPTool",
    "ToolOption",
    "NLPRequest",
    "NLPResponse",
    "get_tool_registry",
    "get_tool_by_id",
    "validate_nlp_request",
    "execute_nlp_tool",
    "register_tool_handler",
    "unregister_tool_handler",
    "is_tool_connected",
    "analyze_text",
    "detect_language",
    "tokenize_text",
    "split_sentences",
    "count_words",
    "count_sentences",
    "stem_and_lemmatize",
    "pos_tag_text",
    "get_text_embedding",
    "classify_text",
    "run_pos_embeddings_classification",
    "compute_tfidf",
    "NGramLanguageModel",
    "calculate_sequence_probability",
    "run_ngram_tool",
    "LatentSemanticIndexing",
    "run_lsi_analysis",
    "run_lsi_tool",
]

