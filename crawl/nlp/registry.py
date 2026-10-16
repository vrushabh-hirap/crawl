"""Registry of supported NLP Lab tools and their options."""

from typing import Dict, List, Optional
from crawl.nlp.schemas import NLPTool, ToolOption

# Catalog of the 6 core NLP Lab tools
_TOOLS: List[NLPTool] = [
    NLPTool(
        id="text_analysis",
        name="Text Analysis",
        category="Linguistic Statistics",
        description="Detects document language, calculates word, sentence, and character counts, and produces cleaned token streams.",
        icon="📊",
        badge="Core Analytics",
        status="available",
        options=[
            ToolOption(
                name="lowercase",
                type="boolean",
                label="Lowercase Tokens",
                description="Converts all extracted tokens to lowercase.",
                default=False,
            ),
            ToolOption(
                name="strip_punctuation",
                type="boolean",
                label="Strip Punctuation",
                description="Removes standard punctuation characters from the token stream.",
                default=False,
            ),
        ],
        sample_text=(
            "Natural Language Processing enables computers to understand human language. "
            "It combines computational linguistics with statistical modeling, machine learning, and deep learning. "
            "How effectively can an algorithm parse syntax and semantic context? Let's analyze this sample!"
        )
    ),
    NLPTool(
        id="stem_lemma",
        name="Stemming & Lemmatization",
        category="Morphological Normalization",
        description="Reduces inflected and derived words to base root forms or dictionary lemmas, featuring a WordNet lemmatizer option.",
        icon="🌱",
        badge="Morphology",
        status="available",
        options=[
            ToolOption(
                name="method",
                type="select",
                label="Normalization Method",
                description="Choose algorithm: WordNet Lemmatizer, spaCy Lemmatizer, or Porter Stemmer.",
                default="wordnet",
                choices=["wordnet", "spacy", "porter", "snowball"],
            ),
            ToolOption(
                name="pos",
                type="select",
                label="WordNet POS Tag",
                description="Part of speech filter for WordNet lemmatization.",
                default="all",
                choices=["all", "n", "v", "a", "r"],
            ),
        ],
        sample_text=(
            "The experienced software engineers were developing distributed web applications, "
            "running comprehensive test suites across multiple universities, and consistently improving API performance."
        )
    ),
    NLPTool(
        id="pos_embeddings_classification",
        name="POS, Embeddings & Classification",
        category="Syntax & Semantics",
        description="Performs grammatical Part-Of-Speech tagging, dense vector embedding representations, and text categorization.",
        icon="🏷️",
        badge="Syntax & Vectors",
        status="available",
        options=[
            ToolOption(
                name="task",
                type="select",
                label="Operation Mode",
                description="Select linguistic task to perform.",
                default="pos_tagging",
                choices=["pos_tagging", "embeddings", "classification"],
            ),
            ToolOption(
                name="candidate_labels",
                type="text",
                label="Classification Labels",
                description="Comma-separated candidate categories for classification.",
                default="Technology, Engineering, Science, Business, General",
            ),
        ],
        sample_text=(
            "The senior backend engineer deployed high-throughput microservices using Docker and Kubernetes on cloud clusters."
        )
    ),
    NLPTool(
        id="tfidf",
        name="TF-IDF Analysis",
        category="Information Retrieval",
        description="Computes Term Frequency - Inverse Document Frequency scores to identify salient vocabulary and keywords across document collections.",
        icon="📈",
        badge="Vector Space",
        status="available",
        options=[
            ToolOption(
                name="ngram_range",
                type="select",
                label="N-Gram Range",
                description="Range of n-gram tokens to include in TF-IDF matrix.",
                default="1-1",
                choices=["1-1", "1-2", "1-3"],
            ),
            ToolOption(
                name="max_features",
                type="number",
                label="Top Keywords Limit",
                description="Maximum number of top-weighted vocabulary terms to display.",
                default=20,
            ),
            ToolOption(
                name="remove_stopwords",
                type="boolean",
                label="Remove Stopwords",
                description="Filter common English stopwords to highlight distinctive terms.",
                default=True,
            ),
            ToolOption(
                name="corpus",
                type="text",
                label="Additional Reference Documents (Optional)",
                description="Separate additional corpus documents with double line breaks.",
                default="",
            ),
        ],
        sample_text=(
            "Information retrieval systems leverage term frequency and inverse document frequency to measure keyword significance.\n\n"
            "TF-IDF diminishes the weight of universally common words while emphasizing terms with high discriminatory power.\n\n"
            "Search engines rely on vector space representations to match search queries against massive document indices."
        )
    ),
    NLPTool(
        id="ngram_probability",
        name="N-gram Sequence Probability",
        category="Statistical Language Modeling",
        description="Estimates statistical language model probabilities for n-grams and computes total sequence probability and perplexity.",
        icon="🎲",
        badge="Language Modeling",
        status="available",
        options=[
            ToolOption(
                name="n",
                type="select",
                label="N-Gram Order",
                description="Order of the n-gram language model (1: Unigram, 2: Bigram, 3: Trigram).",
                default=2,
                choices=[1, 2, 3],
            ),
            ToolOption(
                name="smoothing",
                type="select",
                label="Smoothing Method",
                description="Probability smoothing technique for unseen sequences.",
                default="laplace",
                choices=["laplace", "none"],
            ),
            ToolOption(
                name="target_sequence",
                type="text",
                label="Target Test Sequence (Optional)",
                description="Specific phrase or sentence to evaluate probability against the training text.",
                default="to be or not to be",
            ),
        ],
        sample_text=(
            "to be or not to be that is the question whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune "
            "or to take arms against a sea of troubles and by opposing end them"
        )
    ),
    NLPTool(
        id="lsi_search",
        name="LSI Search & Categorization",
        category="Latent Semantic Analysis",
        description="Applies Latent Semantic Indexing (Truncated SVD) on term-document matrices to discover latent concepts and rank documents by semantic similarity.",
        icon="🔍",
        badge="Dimensionality Reduction",
        status="available",
        options=[
            ToolOption(
                name="mode",
                type="select",
                label="LSI Mode",
                description="Search relevance against query or cluster documents into latent categories.",
                default="search",
                choices=["search", "categorize"],
            ),
            ToolOption(
                name="query",
                type="text",
                label="Search Query",
                description="Query string for semantic document retrieval.",
                default="scalable backend microservices",
            ),
            ToolOption(
                name="num_topics",
                type="number",
                label="Latent Topics (k)",
                description="Number of singular value decomposition components / latent topics.",
                default=2,
            ),
            ToolOption(
                name="show_coordinates",
                type="boolean",
                label="Show Latent Coordinates",
                description="Include document coordinate projections in reduced dimensional space.",
                default=True,
            ),
        ],
        sample_text=(
            "Document 1: Python web development, Flask REST APIs, and PostgreSQL database queries.\n\n"
            "Document 2: Scalable Python backend engineering, Docker microservices architecture, and Kubernetes.\n\n"
            "Document 3: Deep learning neural networks, computer vision, and transformer models in PyTorch.\n\n"
            "Document 4: Machine learning model training, transformers, and deep neural networks in TensorFlow."
        )
    ),
]

_TOOL_MAP: Dict[str, NLPTool] = {tool.id: tool for tool in _TOOLS}


def get_tool_registry() -> List[NLPTool]:
    """Returns list of all registered NLP tools."""
    return list(_TOOLS)


def get_tool_by_id(tool_id: str) -> Optional[NLPTool]:
    """Retrieves an NLP tool definition by its unique identifier."""
    return _TOOL_MAP.get(tool_id)
