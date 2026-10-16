"""Unit tests for NLP Lab service, tool registry, validation, and web routes."""

import json
import pytest
from crawl.nlp import (
    get_tool_registry,
    get_tool_by_id,
    validate_nlp_request,
    execute_nlp_tool,
    register_tool_handler,
    unregister_tool_handler,
    is_tool_connected,
)
from crawl.jobs.web.server import app


# ─────────────────────────────────────────────────────────────────────────────
# 1. NLP Tool Registry Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_nlp_tool_registry_contains_required_tools():
    """Verifies that all 6 required NLP tools are defined and registered."""
    tools = get_tool_registry()
    tool_ids = {t.id for t in tools}

    expected_ids = {
        "text_analysis",
        "stem_lemma",
        "pos_embeddings_classification",
        "tfidf",
        "ngram_probability",
        "lsi_search",
    }
    assert expected_ids.issubset(tool_ids), f"Missing tools in registry: {expected_ids - tool_ids}"


def test_tool_options_and_metadata():
    """Verifies tool options structure, including WordNet option on stem_lemma."""
    stem_tool = get_tool_by_id("stem_lemma")
    assert stem_tool is not None
    method_opt = next((opt for opt in stem_tool.options if opt.name == "method"), None)
    assert method_opt is not None
    assert "wordnet" in method_opt.choices
    assert method_opt.default == "wordnet"

    ngram_tool = get_tool_by_id("ngram_probability")
    assert ngram_tool is not None
    n_opt = next((opt for opt in ngram_tool.options if opt.name == "n"), None)
    assert n_opt is not None
    assert 2 in n_opt.choices or "2" in [str(c) for c in n_opt.choices]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Input Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_validation_rejects_empty_text():
    """Verifies that empty or whitespace-only text is rejected."""
    is_valid, err, _ = validate_nlp_request("text_analysis", "")
    assert not is_valid
    assert "empty" in err.lower()

    is_valid, err, _ = validate_nlp_request("text_analysis", "   \n\t   ")
    assert not is_valid
    assert "empty" in err.lower()


def test_validation_rejects_unknown_tool():
    """Verifies that non-existent tool IDs produce clear errors."""
    is_valid, err, _ = validate_nlp_request("non_existent_tool_123", "Some text")
    assert not is_valid
    assert "unknown nlp tool" in err.lower()


def test_validation_validates_options_choices():
    """Verifies that invalid select choices are caught."""
    is_valid, err, _ = validate_nlp_request("stem_lemma", "Some valid text", {"method": "invalid_stemmer"})
    assert not is_valid
    assert "allowed choices" in err.lower()

    is_valid, err, opts = validate_nlp_request("stem_lemma", "Some valid text", {"method": "wordnet"})
    assert is_valid
    assert err is None
    assert opts["method"] == "wordnet"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Execution & Dynamic Handler Registration Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_execute_pending_tool():
    """Verifies that tools without an active implementation return a clean pending_implementation response."""
    tool_id = "lsi_search"
    from crawl.nlp.service import _TOOL_HANDLERS
    orig_handler = _TOOL_HANDLERS.get(tool_id)
    try:
        unregister_tool_handler(tool_id)
        assert not is_tool_connected(tool_id)

        response = execute_nlp_tool(tool_id, "Sample latent semantic analysis document text", {"num_topics": 2})
        assert response["success"] is True
        assert response["status"] == "pending_implementation"
        assert response["tool_id"] == "lsi_search"
        assert "validation" in response
        assert response["validation"]["text_characters"] > 0
        assert response["validation"]["text_words"] > 0
    finally:
        if orig_handler:
            register_tool_handler(tool_id, orig_handler)


def test_execute_connected_handler():
    """Verifies that registered tool handlers are executed and output is formatted properly."""
    tool_id = "lsi_search"
    from crawl.nlp.service import _TOOL_HANDLERS
    orig_handler = _TOOL_HANDLERS.get(tool_id)

    def mock_lsi_handler(text: str, options: dict) -> dict:
        return {"topics": [("topic_1", 0.95)]}

    try:
        register_tool_handler(tool_id, mock_lsi_handler)
        assert is_tool_connected(tool_id)

        response = execute_nlp_tool(tool_id, "test sentence")
        assert response["success"] is True
        assert response["status"] == "completed"
        assert response["result"]["topics"] == [("topic_1", 0.95)]

    finally:
        if orig_handler:
            register_tool_handler(tool_id, orig_handler)
        else:
            unregister_tool_handler(tool_id)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Text Analysis Engine Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_text_analysis_english():
    """Tests text analysis on standard English text."""
    from crawl.nlp import analyze_text

    text = "Natural Language Processing enables computers to understand human language. It is a vital branch of AI."
    result = analyze_text(text)

    assert result["language"] == "en"
    assert result["language_name"] == "English"
    assert result["is_uncertain"] is False
    assert result["language_status"] == "detected"
    assert result["language_confidence"] > 0.5
    assert result["sentence_count"] == 2
    assert result["word_count"] >= 15
    assert len(result["tokens"]) > 0
    assert "Natural" in result["tokens"]


def test_text_analysis_multilingual():
    """Tests language detection across Spanish, French, German, and Russian."""
    from crawl.nlp import analyze_text

    # Spanish
    es_res = analyze_text("El procesamiento del lenguaje natural permite a las computadoras comprender el lenguaje humano.")
    assert es_res["language"] == "es"
    assert es_res["is_uncertain"] is False
    assert es_res["word_count"] > 5

    # French
    fr_res = analyze_text("Le traitement automatique du langage naturel permet aux ordinateurs de comprendre le langage humain.")
    assert fr_res["language"] == "fr"
    assert fr_res["is_uncertain"] is False

    # German
    de_res = analyze_text("Die Verarbeitung natürlicher Sprache ermöglicht es Computern, menschliche Sprache zu verstehen.")
    assert de_res["language"] == "de"
    assert de_res["is_uncertain"] is False

    # Russian (Cyrillic)
    ru_res = analyze_text("Обработка естественного языка позволяет компьютерам понимать человеческий язык.")
    assert ru_res["language"] == "ru"
    assert ru_res["is_uncertain"] is False


def test_text_analysis_empty_and_whitespace():
    """Tests that empty and whitespace-only text is handled gracefully without crashing."""
    from crawl.nlp import analyze_text

    for empty_val in ["", "   ", "\n\t  \n"]:
        res = analyze_text(empty_val)
        assert res["language"] == "unknown"
        assert res["is_uncertain"] is True
        assert res["language_status"] == "unavailable"
        assert res["word_count"] == 0
        assert res["sentence_count"] == 0
        assert res["tokens"] == []
        assert res["tokens_count"] == 0


def test_text_analysis_uncertain_short_text():
    """Tests that very short text triggers a clear uncertainty indication."""
    from crawl.nlp import analyze_text

    res = analyze_text("Hello")
    assert res["is_uncertain"] is True
    assert res["language_status"] == "uncertain"
    assert res["language_message"] is not None
    assert res["word_count"] == 1
    assert res["sentence_count"] == 1
    assert res["tokens"] == ["Hello"]


def test_text_analysis_unsupported_language_or_gibberish():
    """Tests that unrecognizable random characters or unsupported languages are handled gracefully."""
    from crawl.nlp import analyze_text

    res = analyze_text("xkjdfh skdjfh lksjdfh qowieur zxncbv")
    assert res["language"] == "unknown"
    assert res["is_uncertain"] is True
    assert res["language_status"] in ("unsupported", "uncertain")
    assert res["word_count"] == 5
    assert len(res["tokens"]) == 5


def test_text_analysis_token_options():
    """Tests options for lowercase conversion and punctuation stripping."""
    from crawl.nlp import analyze_text

    text = "Hello, World! Here is a state-of-the-art model."
    res_default = analyze_text(text)
    assert "," in res_default["tokens"]
    assert "Hello" in res_default["tokens"]

    res_clean = analyze_text(text, options={"lowercase": True, "strip_punctuation": True})
    assert "," not in res_clean["tokens"]
    assert "!" not in res_clean["tokens"]
    assert "hello" in res_clean["tokens"]
    assert "world" in res_clean["tokens"]


def test_text_analysis_sentence_splitting_abbreviations():
    """Tests abbreviation-aware sentence splitting."""
    from crawl.nlp import split_sentences, count_sentences

    text = "Dr. Smith bought 3.14 lbs of apples from the U.S. market! Did he pay? Yes, he did."
    sents = split_sentences(text)
    assert len(sents) == 3
    assert count_sentences(text) == 3
    assert "Dr. Smith" in sents[0]



# ─────────────────────────────────────────────────────────────────────────────
# 4. Web Route / API Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_nlp_tools_endpoint(client):
    """Verifies GET /api/nlp/tools returns list of tools with metadata."""
    res = client.get("/api/nlp/tools")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "tools" in data
    assert len(data["tools"]) == 6
    names = [t["name"] for t in data["tools"]]
    assert "Text Analysis" in names
    assert "Stemming & Lemmatization" in names


def test_api_nlp_run_empty_text_error(client):
    """Verifies POST /api/nlp/run returns 400 when text is empty."""
    res = client.post(
        "/api/nlp/run",
        json={"tool_id": "text_analysis", "text": "   "},
    )
    assert res.status_code == 400
    data = res.get_json()
    assert data["success"] is False
    assert "empty" in data["error"].lower()


def test_api_nlp_run_missing_tool_error(client):
    """Verifies POST /api/nlp/run returns 400 when tool_id is missing or invalid."""
    res = client.post(
        "/api/nlp/run",
        json={"text": "valid text"},
    )
    assert res.status_code == 400
    data = res.get_json()
    assert data["success"] is False
    assert "tool_id" in data["error"].lower()

    res = client.post(
        "/api/nlp/run",
        json={"tool_id": "unknown_tool_xyz", "text": "valid text"},
    )
    assert res.status_code == 400
    data = res.get_json()
    assert data["success"] is False
    assert "unknown nlp tool" in data["error"].lower()


def test_api_nlp_run_valid_pending_tool(client):
    """Verifies POST /api/nlp/run returns 200 with pending status when a tool handler is pending."""
    tool_id = "lsi_search"
    from crawl.nlp.service import _TOOL_HANDLERS
    orig_handler = _TOOL_HANDLERS.get(tool_id)
    try:
        unregister_tool_handler(tool_id)
        res = client.post(
            "/api/nlp/run",
            json={
                "tool_id": tool_id,
                "text": "Document 1 content\n\nDocument 2 content",
                "options": {"mode": "search", "query": "python", "num_topics": 2},
            },
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["tool_id"] == tool_id
        assert data["status"] == "pending_implementation"
        assert "validation" in data
        assert data["validation"]["options"]["mode"] == "search"
    finally:
        if orig_handler:
            register_tool_handler(tool_id, orig_handler)


def test_api_nlp_execute_alias(client):
    """Verifies POST /api/nlp/execute works as an alias for /api/nlp/run."""
    res = client.post(
        "/api/nlp/execute",
        json={
            "tool_id": "ngram_probability",
            "text": "The quick brown fox jumps over the lazy dog",
            "options": {"n": 2, "smoothing": "laplace"},
        },
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["tool_id"] == "ngram_probability"


def test_api_nlp_run_text_analysis_connected(client):
    """Verifies POST /api/nlp/run executes text_analysis and returns completed results with all fields."""
    res = client.post(
        "/api/nlp/run",
        json={
            "tool_id": "text_analysis",
            "text": "Natural Language Processing allows machines to analyze human sentences effectively. This is awesome!",
            "options": {"lowercase": True, "strip_punctuation": True},
        },
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["tool_id"] == "text_analysis"
    assert data["status"] == "completed"
    assert "result" in data
    result = data["result"]
    assert result["language"] == "en"
    assert result["language_name"] == "English"
    assert result["is_uncertain"] is False
    assert result["word_count"] > 10
    assert result["sentence_count"] == 2
    assert isinstance(result["tokens"], list)
    assert len(result["tokens"]) > 0
    # Verified options: lowercase and strip_punctuation
    assert "natural" in result["tokens"]
    assert "." not in result["tokens"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Stemming & WordNet Lemmatization Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_stem_lemma_wordnet_primary():
    """Verifies that WordNet lemmatization works as primary method with correct method_used attribution."""
    from crawl.nlp.tools.stem_lemma import stem_and_lemmatize

    sample = "The engineers were developing distributed web applications and universities."
    res = stem_and_lemmatize(sample, {"method": "wordnet", "pos": "all"})

    assert res["method_requested"] == "wordnet"
    assert res["method_used"] == "wordnet"
    assert res["is_fallback"] is False
    assert res["warning"] is None
    assert "WordNet" in res["algorithm"]

    # Verify expected lemmas
    assert "engineer" in res["tokens"]
    assert "develop" in res["tokens"]
    assert "application" in res["tokens"]
    assert "university" in res["tokens"]

    # Verify item-level tracking
    items_by_orig = {item["original"].lower(): item for item in res["items"]}
    assert items_by_orig["universities"]["normalized"] == "university"
    assert items_by_orig["universities"]["changed"] is True
    assert items_by_orig["developing"]["normalized"] == "develop"


def test_stem_lemma_wordnet_pos_filtering():
    """Verifies POS tag filtering behavior for WordNet lemmatization."""
    from crawl.nlp.tools.stem_lemma import stem_and_lemmatize

    text = "The meeting was meeting expectations."
    res_v = stem_and_lemmatize(text, {"method": "wordnet", "pos": "v"})
    assert res_v["method_used"] == "wordnet"
    # Under verb POS, 'meeting' reduces to 'meet'
    assert "meet" in res_v["tokens"]

    res_n = stem_and_lemmatize(text, {"method": "wordnet", "pos": "n"})
    assert res_n["method_used"] == "wordnet"
    # Under noun POS, 'meeting' remains 'meeting'
    assert "meeting" in res_n["tokens"]


def test_stem_lemma_spacy_and_porter_methods():
    """Verifies that spaCy and Porter stemmer methods identify their actual method_used."""
    from crawl.nlp.tools.stem_lemma import stem_and_lemmatize

    text = "The software engineers were developing distributed web applications."

    # spaCy Lemmatizer
    spacy_res = stem_and_lemmatize(text, {"method": "spacy"})
    assert spacy_res["method_used"] == "spacy"
    assert spacy_res["is_fallback"] is False
    assert "engineer" in spacy_res["tokens"]
    assert "develop" in spacy_res["tokens"]

    # Porter Stemmer
    porter_res = stem_and_lemmatize(text, {"method": "porter"})
    assert porter_res["method_used"] == "porter"
    assert porter_res["is_fallback"] is False
    assert "engin" in porter_res["tokens"]
    assert "develop" in porter_res["tokens"]

    # Snowball Stemmer
    snowball_res = stem_and_lemmatize(text, {"method": "snowball"})
    assert snowball_res["method_used"] == "snowball"
    assert snowball_res["is_fallback"] is False


def test_stem_lemma_wordnet_missing_instructions_error():
    """Verifies that missing WordNet data raises LookupError with clear installation instructions."""
    import crawl.nlp.tools.stem_lemma as sl

    orig_loader = sl.get_wordnet_lemmatizer
    try:
        sl.get_wordnet_lemmatizer = lambda: None

        with pytest.raises(LookupError) as exc_info:
            sl.stem_and_lemmatize("Some text to test", {"method": "wordnet", "fallback_on_missing": False})

        err_msg = str(exc_info.value)
        assert "Required NLTK WordNet data is missing" in err_msg
        assert "python -m nltk.downloader wordnet omw-1.4" in err_msg
        assert "nltk.download('wordnet')" in err_msg
    finally:
        sl.get_wordnet_lemmatizer = orig_loader


def test_stem_lemma_wordnet_fallback_never_claims_wordnet():
    """Verifies that fallback NEVER silently claims WordNet was used."""
    import crawl.nlp.tools.stem_lemma as sl

    orig_loader = sl.get_wordnet_lemmatizer
    try:
        sl.get_wordnet_lemmatizer = lambda: None

        res = sl.stem_and_lemmatize("Testing fallback behavior", {"method": "wordnet", "fallback_on_missing": True})
        assert res["is_fallback"] is True
        assert res["method_used"] != "wordnet"
        assert "fallback" in res["method_used"].lower()
        assert res["warning"] is not None
        assert "Required NLTK WordNet data is missing" in res["warning"]
    finally:
        sl.get_wordnet_lemmatizer = orig_loader


def test_stage3_normalizer_wordnet():
    """Verifies that Stage 3 Normalizer supports WordNet and identifies method_used."""
    from crawl.jobs.ats.pipeline.stage3_normalizer import normalize, NormalizerConfig

    text = "The engineers were developing distributed web applications and universities."
    config = NormalizerConfig(method="wordnet", wordnet_pos="all", remove_stopwords=False)
    output = normalize(text, config)

    assert output.method_used == "wordnet"
    assert output.warning is None
    assert "engineer" in output.tokens
    assert "develop" in output.tokens
    assert "university" in output.tokens


def test_stage3_normalizer_wordnet_missing_handling():
    """Verifies Stage 3 Normalizer handling when WordNet is missing."""
    import crawl.jobs.ats.pipeline.stage3_normalizer as sn

    orig_loader = sn._load_wordnet
    try:
        sn._load_wordnet = lambda: None

        # 1. raise_on_missing_wordnet=True
        cfg_raise = sn.NormalizerConfig(method="wordnet", raise_on_missing_wordnet=True)
        with pytest.raises(LookupError) as exc_info:
            sn.normalize("Developing applications", cfg_raise)
        assert "Required NLTK WordNet data is missing" in str(exc_info.value)

        # 2. raise_on_missing_wordnet=False
        cfg_fallback = sn.NormalizerConfig(method="wordnet", raise_on_missing_wordnet=False)
        out = sn.normalize("Developing applications", cfg_fallback)
        assert out.method_used != "wordnet"
        assert "fallback" in out.method_used.lower()
        assert out.warning is not None
        assert "python -m nltk.downloader wordnet" in out.warning
    finally:
        sn._load_wordnet = orig_loader


def test_api_nlp_run_stem_lemma_endpoints(client):
    """Verifies POST /api/nlp/run executes stem_lemma with wordnet, spacy, and porter."""
    # 1. WordNet
    res_wn = client.post(
        "/api/nlp/run",
        json={
            "tool_id": "stem_lemma",
            "text": "The senior engineers were developing scalable web applications and testing universities.",
            "options": {"method": "wordnet", "pos": "all"},
        },
    )
    assert res_wn.status_code == 200
    data_wn = res_wn.get_json()
    assert data_wn["success"] is True
    assert data_wn["status"] == "completed"
    assert data_wn["result"]["method_used"] == "wordnet"
    assert data_wn["result"]["is_fallback"] is False
    assert "engineer" in data_wn["result"]["tokens"]
    assert "university" in data_wn["result"]["tokens"]

    # 2. spaCy
    res_sp = client.post(
        "/api/nlp/run",
        json={
            "tool_id": "stem_lemma",
            "text": "The senior engineers were developing scalable web applications.",
            "options": {"method": "spacy"},
        },
    )
    assert res_sp.status_code == 200
    data_sp = res_sp.get_json()
    assert data_sp["success"] is True
    assert data_sp["result"]["method_used"] == "spacy"

    # 3. Porter
    res_pt = client.post(
        "/api/nlp/run",
        json={
            "tool_id": "stem_lemma",
            "text": "The senior engineers were developing scalable web applications.",
            "options": {"method": "porter"},
        },
    )
    assert res_pt.status_code == 200
    data_pt = res_pt.get_json()
    assert data_pt["success"] is True
    assert data_pt["result"]["method_used"] == "porter"


# ─────────────────────────────────────────────────────────────────────────────
# 6. POS Tagging, Embeddings & Classification Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_pos_tagging_operation():
    """Verifies POS tagging returns tokens, tags, descriptions, and distributions."""
    from crawl.nlp import pos_tag_text

    text = "The senior software engineer deployed Kubernetes microservices on AWS."
    res = pos_tag_text(text)

    assert res["task"] == "pos_tagging"
    assert res["tokens_count"] >= 9
    assert len(res["tagged_tokens"]) == res["tokens_count"]

    first = res["tagged_tokens"][0]
    assert "token" in first
    assert "pos" in first
    assert "tag" in first
    assert "description" in first
    assert len(first["description"]) > 0

    assert "NOUN" in res["pos_distribution"] or "PROPN" in res["pos_distribution"]
    assert res["lexical_density"] > 0.0

    # Empty text
    empty_res = pos_tag_text("")
    assert empty_res["tokens_count"] == 0
    assert empty_res["tagged_tokens"] == []


def test_text_embeddings_operation():
    """Verifies dense embedding generation, dimensions, stats, and fallback notice."""
    from crawl.nlp import get_text_embedding

    text = "Deploying high-throughput distributed pipelines on cloud clusters."
    res = get_text_embedding(text)

    assert res["task"] == "embeddings"
    assert res["dimensions"] in (96, 384)
    assert len(res["vector"]) == res["dimensions"]
    assert len(res["vector_preview"]) <= 16
    assert res["vector_norm"] > 0.0
    assert "stats" in res
    assert "notice" in res
    assert len(res["notice"]) > 0

    # Test empty input
    empty_res = get_text_embedding("")
    assert empty_res["dimensions"] == 0
    assert empty_res["vector"] == []


def test_text_classification_benchmark():
    """Verifies classification on documented benchmark corpus with explanations."""
    from crawl.nlp import classify_text

    # Technology text
    tech_text = "Senior Python backend developer building Docker container microservices and Redis caching."
    tech_res = classify_text(tech_text)
    assert tech_res["task"] == "classification"
    assert tech_res["predicted_label"] == "Technology"
    assert tech_res["confidence"] > 0.5
    assert len(tech_res["ranking"]) >= 3
    assert tech_res["ranking"][0]["label"] == "Technology"
    assert "evaluation" in tech_res
    assert tech_res["evaluation"]["benchmark_accuracy"] == 1.0
    assert "explanation" in tech_res

    # Business text
    biz_text = "Quarterly financial accounting, revenue growth forecast, and venture capital budgeting."
    biz_res = classify_text(biz_text)
    assert biz_res["predicted_label"] == "Business"

    # Science text
    sci_text = "Genomic DNA sequencing and molecular biology laboratory research."
    sci_res = classify_text(sci_text)
    assert sci_res["predicted_label"] == "Science"


def test_text_classification_user_provided_dataset():
    """Verifies classification using a custom user-provided dataset."""
    from crawl.nlp import classify_text

    custom_dataset = {
        "Sports": [
            "Soccer football match championship goal striker stadium referee.",
            "Basketball tournament playoffs points rebounds court slam dunk.",
        ],
        "Cooking": [
            "Italian pasta recipe tomato basil garlic olive oil dinner cooking.",
            "Baking chocolate cake dessert sugar flour oven pastry delicious.",
        ],
    }

    input_text = "The championship soccer striker scored an amazing goal in the stadium."
    res = classify_text(input_text, dataset=custom_dataset)

    assert res["predicted_label"] == "Sports"
    assert res["confidence"] > 0.5
    assert res["evaluation"]["dataset_type"] == "user_provided"
    assert "Sports" in res["evaluation"]["training_categories"]
    assert "Cooking" in res["evaluation"]["training_categories"]


def test_api_nlp_run_pos_embeddings_classification_endpoints(client):
    """Verifies POST /api/nlp/run executes all tasks of pos_embeddings_classification."""
    # 1. POS Tagging
    res_pos = client.post(
        "/api/nlp/run",
        json={
            "tool_id": "pos_embeddings_classification",
            "text": "The senior engineer built scalable web applications.",
            "options": {"task": "pos_tagging"},
        },
    )
    assert res_pos.status_code == 200
    data_pos = res_pos.get_json()
    assert data_pos["success"] is True
    assert data_pos["status"] == "completed"
    assert data_pos["result"]["task"] == "pos_tagging"
    assert len(data_pos["result"]["tagged_tokens"]) > 0

    # 2. Embeddings
    res_emb = client.post(
        "/api/nlp/run",
        json={
            "tool_id": "pos_embeddings_classification",
            "text": "The senior engineer built scalable web applications.",
            "options": {"task": "embeddings"},
        },
    )
    assert res_emb.status_code == 200
    data_emb = res_emb.get_json()
    assert data_emb["success"] is True
    assert data_emb["result"]["task"] == "embeddings"
    assert len(data_emb["result"]["vector"]) in (96, 384)

    # 3. Classification
    res_cls = client.post(
        "/api/nlp/run",
        json={
            "tool_id": "pos_embeddings_classification",
            "text": "Senior engineer deploying microservices with Docker on AWS.",
            "options": {"task": "classification"},
        },
    )
    assert res_cls.status_code == 200
    data_cls = res_cls.get_json()
    assert data_cls["success"] is True
    assert data_cls["result"]["task"] == "classification"
    assert data_cls["result"]["predicted_label"] == "Technology"


# ─────────────────────────────────────────────────────────────────────────────
# 7. TF-IDF Analysis Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_tfidf_multi_documents():
    """Verifies TF-IDF vocabulary, document matrix, and distinctive term discovery."""
    from crawl.nlp import compute_tfidf

    docs = (
        "Information retrieval systems leverage term frequency and inverse document frequency to measure keyword significance.\n\n"
        "TF-IDF diminishes the weight of universally common words while emphasizing terms with high discriminatory power.\n\n"
        "Search engines rely on vector space representations to match search queries against massive document indices."
    )

    res = compute_tfidf(docs, {"ngram_range": "1-1", "max_features": 15})

    assert res["num_documents"] == 3
    assert res["vocabulary_size"] >= 20
    assert len(res["documents"]) == 3
    assert len(res["matrix"]) <= 15
    assert len(res["distinguishing_analysis"]) == 3

    # Check that Doc 1's top terms contain 'frequency'
    doc1_terms = [t["term"] for t in res["documents"][0]["top_terms"]]
    assert "frequency" in doc1_terms

    # Check that Doc 2's top terms contain 'tf-idf' or 'weight'
    doc2_terms = [t["term"] for t in res["documents"][1]["top_terms"]]
    assert "tf-idf" in doc2_terms or "weight" in doc2_terms

    # Check that Doc 3's top terms contain 'search'
    doc3_terms = [t["term"] for t in res["documents"][2]["top_terms"]]
    assert "search" in doc3_terms

    # Distinguishing analysis checks
    assert any("Doc 1" in s for s in res["distinguishing_analysis"])
    assert any("Doc 2" in s for s in res["distinguishing_analysis"])
    assert any("Doc 3" in s for s in res["distinguishing_analysis"])


def test_compute_tfidf_ngrams_and_features():
    """Verifies n-gram range and max_features handling in TF-IDF."""
    from crawl.nlp import compute_tfidf

    docs = [
        "Machine learning models analyze structured dataset features.",
        "Deep neural network models train on massive unstructured dataset images.",
    ]

    # Unigrams + Bigrams
    res_bigram = compute_tfidf(docs, {"ngram_range": "1-2", "max_features": 25})
    assert res_bigram["num_documents"] == 2
    # Verify bigrams exist in vocabulary
    has_bigram = any(" " in term for term in res_bigram["vocabulary"])
    assert has_bigram

    # max_features cap
    res_capped = compute_tfidf(docs, {"ngram_range": "1-1", "max_features": 5})
    assert len(res_capped["matrix"]) <= 5


def test_compute_tfidf_empty():
    """Verifies empty input handling for TF-IDF."""
    from crawl.nlp import compute_tfidf

    res = compute_tfidf("")
    assert res["num_documents"] == 0
    assert res["vocabulary_size"] == 0
    assert res["matrix"] == []


def test_api_nlp_run_tfidf_endpoint(client):
    """Verifies POST /api/nlp/run executes tfidf and returns completed results."""
    docs = (
        "Python backend microservices using Flask and Docker.\n\n"
        "Machine learning models using PyTorch and Scikit-learn.\n\n"
        "Cloud infrastructure deployment on AWS with Kubernetes."
    )

    res = client.post(
        "/api/nlp/run",
        json={
            "tool_id": "tfidf",
            "text": docs,
            "options": {"ngram_range": "1-1", "max_features": 10},
        },
    )

    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["status"] == "completed"
    assert data["result"]["num_documents"] == 3
    assert len(data["result"]["documents"]) == 3
    assert len(data["result"]["matrix"]) <= 10
    assert len(data["result"]["distinguishing_analysis"]) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 9. N-Gram Language Model Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_ngram_model_orders_and_probabilities():
    """Verifies unigram, bigram, and trigram language models calculate valid probabilities and perplexities."""
    from crawl.nlp import NGramLanguageModel, calculate_sequence_probability

    corpus = (
        "to be or not to be that is the question "
        "whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune"
    )

    # 1. Bigram Model (n=2)
    bigram_res = calculate_sequence_probability(
        corpus=corpus,
        target_sequence="to be or not to be",
        n=2,
        smoothing="laplace",
    )
    assert bigram_res["n"] == 2
    assert bigram_res["order_name"] == "Bigram"
    assert bigram_res["total_probability"] > 0.0
    assert bigram_res["perplexity"] > 1.0
    assert len(bigram_res["steps"]) == len(bigram_res["target_tokens"])
    assert bigram_res["unseen_ngram_count"] == 0  # this exact sequence appeared in corpus

    # Check step details
    first_step = bigram_res["steps"][0]
    assert first_step["target_word"] == "to"
    assert first_step["context"] == ["<s>"]
    assert "P(to | <s>)" in first_step["formula_str"]
    assert first_step["status"] == "seen"

    # 2. Trigram Model (n=3)
    trigram_res = calculate_sequence_probability(
        corpus=corpus,
        target_sequence="to be or not to be",
        n=3,
        smoothing="laplace",
    )
    assert trigram_res["n"] == 3
    assert trigram_res["order_name"] == "Trigram"
    assert trigram_res["total_probability"] > 0.0
    assert trigram_res["perplexity"] > 1.0

    # 3. Direct Class Instance Fit & Evaluate
    model = NGramLanguageModel(n=2, smoothing="laplace").fit(corpus)
    res_direct = model.calculate_sequence_probability("to be")
    assert res_direct["total_probability"] > 0.0
    assert res_direct["target_tokens"] == ["to", "be"]


def test_ngram_unseen_sequence_handling_laplace_vs_none():
    """Verifies that unsmoothed MLE collapses to 0 on unseen sequences, whereas Laplace smoothing prevents it."""
    from crawl.nlp import calculate_sequence_probability

    corpus = "the quick brown fox jumps over the lazy dog"
    # "quantum physics computer" never appeared in the corpus
    unseen_phrase = "quantum physics computer"

    # 1. Unsmoothed MLE (smoothing="none")
    mle_res = calculate_sequence_probability(
        corpus=corpus,
        target_sequence=unseen_phrase,
        n=2,
        smoothing="none",
    )
    assert mle_res["smoothing"] == "none"
    assert mle_res["total_probability"] == 0.0
    assert mle_res["total_probability_scientific"] == "0.0"
    assert mle_res["perplexity"] in (float("inf"), "Infinity")
    assert mle_res["unseen_ngram_count"] > 0
    # Explanation should highlight zero probability collapse
    assert "0.0" in mle_res["explanation"] or "zero" in mle_res["explanation"].lower()

    # 2. Laplace (+1) Smoothing (smoothing="laplace")
    laplace_res = calculate_sequence_probability(
        corpus=corpus,
        target_sequence=unseen_phrase,
        n=2,
        smoothing="laplace",
    )
    assert laplace_res["smoothing"] == "laplace"
    # Crucial property: probability MUST NOT be zero!
    assert laplace_res["total_probability"] > 0.0
    assert laplace_res["total_probability_scientific"] != "0.0"
    assert laplace_res["perplexity"] not in (float("inf"), "Infinity")
    assert laplace_res["perplexity"] > 1.0
    # Unseen status flags should be present on the steps
    has_unseen_step = any(s["is_unseen"] for s in laplace_res["steps"])
    assert has_unseen_step
    # Formula should document the added +1 / +V
    assert "+ 1.0" in laplace_res["steps"][0]["formula_str"] or "+ 1" in laplace_res["steps"][0]["formula_str"]


def test_ngram_add_k_smoothing():
    """Verifies Add-k smoothing with arbitrary k factor."""
    from crawl.nlp import calculate_sequence_probability

    corpus = "artificial intelligence and machine learning models"
    target = "deep learning models"

    res_k = calculate_sequence_probability(
        corpus=corpus,
        target_sequence=target,
        n=2,
        smoothing="add_k",
        k=0.5,
    )
    assert res_k["smoothing"] == "add_k"
    assert res_k["smoothing_k"] == 0.5
    assert res_k["total_probability"] > 0.0
    assert "+ 0.5" in res_k["steps"][0]["formula_str"]


def test_ngram_empty_inputs_graceful():
    """Verifies that empty corpus or empty sequence returns informative results without crashing."""
    from crawl.nlp import calculate_sequence_probability

    # Empty target sequence
    res_empty_seq = calculate_sequence_probability("some valid text", "")
    assert res_empty_seq["total_probability"] == 0.0
    assert res_empty_seq["steps"] == []
    assert "empty" in res_empty_seq["explanation"].lower()

    # Empty corpus
    res_empty_corpus = calculate_sequence_probability("", "target words")
    assert res_empty_corpus["total_probability"] == 0.0
    assert "empty" in res_empty_corpus["explanation"].lower()


def test_api_nlp_run_ngram_endpoint(client):
    """Verifies POST /api/nlp/run executes ngram_probability and returns completed explainable results."""
    corpus = (
        "to be or not to be that is the question "
        "whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune"
    )

    res = client.post(
        "/api/nlp/run",
        json={
            "tool_id": "ngram_probability",
            "text": corpus,
            "options": {
                "n": 2,
                "smoothing": "laplace",
                "target_sequence": "to be or not to be",
            },
        },
    )

    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["status"] == "completed"
    assert data["result"]["n"] == 2
    assert data["result"]["order_name"] == "Bigram"
    assert data["result"]["smoothing"] == "laplace"
    assert data["result"]["total_probability"] > 0.0
    assert len(data["result"]["steps"]) == 6
    assert "explanation" in data["result"]
    assert "unseen_handling_summary" in data["result"]


# ─────────────────────────────────────────────────────────────────────────────
# 10. Latent Semantic Indexing (LSI) Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_lsi_search_relevance_ranking():
    """Verifies that LSI projects query and documents into latent space and ranks by cosine similarity."""
    from crawl.nlp import run_lsi_analysis, LatentSemanticIndexing

    documents = [
        "Document 1: Python and Flask web development with relational databases, PostgreSQL, and SQL queries.",
        "Document 2: Deep learning neural networks, computer vision, and transformer models in PyTorch and TensorFlow.",
        "Document 3: Scalable backend engineering, Docker microservices architecture, and Kubernetes orchestration.",
        "Document 4: Natural language processing, tokenization, stemming algorithms, and sentiment analysis.",
    ]

    # Query clearly targeting Doc 3
    query = "scalable backend microservices docker"
    res = run_lsi_analysis(documents, query=query, num_topics=2, mode="search")

    assert res["task"] == "lsi_search"
    assert res["mode"] == "search"
    assert res["num_documents"] == 4
    assert res["num_topics_used"] == 2
    assert len(res["rankings"]) == 4

    # Top ranked document must be Doc 3
    top_doc = res["rankings"][0]
    assert top_doc["doc_id"] == "Doc 3"
    assert top_doc["similarity"] > 0.5
    assert len(top_doc["coordinates"]) == 2

    # Check that topics are populated with singular values and term loadings
    assert len(res["topics"]) == 2
    for topic in res["topics"]:
        assert topic["singular_value"] > 0.0
        assert len(topic["top_terms"]) > 0

    # Also test object-oriented LatentSemanticIndexing
    lsi_model = LatentSemanticIndexing(num_topics=2)
    lsi_model.fit(documents)
    search_results = lsi_model.search("deep learning neural networks")
    assert search_results[0]["doc_id"] == "Doc 2"


def test_lsi_categorization_mode():
    """Verifies that LSI clusters documents into latent topic categories."""
    from crawl.nlp import run_lsi_analysis

    documents = [
        "Python Flask web development and backend APIs.",
        "Django web application development and PostgreSQL.",
        "Deep learning transformers, PyTorch, and neural networks.",
        "Convolutional neural networks for computer vision with TensorFlow.",
    ]

    res = run_lsi_analysis(documents, num_topics=2, mode="categorize")
    assert res["mode"] == "categorize"
    assert len(res["categories"]) == 2

    total_assigned = sum(c["assigned_documents_count"] for c in res["categories"])
    assert total_assigned == 4


def test_lsi_component_and_collection_validation_bounds():
    """Verifies that small inputs, excessive k, single docs, and OOV queries do not crash."""
    from crawl.nlp import run_lsi_analysis

    # 1. Excessive k (requested k=10 for 2 documents)
    two_docs = [
        "Python backend development with Flask.",
        "Machine learning models with PyTorch.",
    ]
    res_k_clamped = run_lsi_analysis(two_docs, num_topics=10)
    # k must be clamped to max rank (min(2, vocab_size))
    assert res_k_clamped["num_topics_used"] <= 2
    assert res_k_clamped["k_adjustment_note"] is not None
    assert "adjusted" in res_k_clamped["k_adjustment_note"].lower()

    # 2. Single document collection (N=1)
    single_doc = "Single lonely document for testing minimum bounds."
    res_single = run_lsi_analysis([single_doc], num_topics=3)
    assert res_single["num_documents"] == 1
    assert res_single["num_topics_used"] == 1
    assert len(res_single["rankings"]) == 1

    # 3. Completely out-of-vocabulary query
    res_oov = run_lsi_analysis(two_docs, query="quantum thermodynamics astrophysics")
    assert len(res_oov["rankings"]) == 2
    # All similarities should be 0.0 safely without division by zero
    for r in res_oov["rankings"]:
        assert r["similarity"] == 0.0

    # 4. Empty document list
    res_empty = run_lsi_analysis([])
    assert res_empty["num_documents"] == 0
    assert res_empty["topics"] == []
    assert "No documents" in res_empty["explanation"]


def test_lsi_label_distinction():
    """Verifies LSI explicitly notes separation from sentence-transformers."""
    from crawl.nlp import run_lsi_analysis

    res = run_lsi_analysis(["Doc 1", "Doc 2"])
    assert "sentence-transformer" in res["lsi_label_notice"].lower()
    assert "svd" in res["lsi_label_notice"].lower()


def test_api_nlp_run_lsi_endpoint(client):
    """Verifies POST /api/nlp/run executes lsi_search and returns completed results."""
    documents = (
        "Document 1: Python and Flask web development with relational databases.\n\n"
        "Document 2: Deep learning neural networks and transformer models in PyTorch.\n\n"
        "Document 3: Scalable backend engineering, Docker microservices architecture, and Kubernetes."
    )

    res = client.post(
        "/api/nlp/run",
        json={
            "tool_id": "lsi_search",
            "text": documents,
            "options": {
                "mode": "search",
                "query": "microservices kubernetes",
                "num_topics": 2,
                "show_coordinates": True,
            },
        },
    )

    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["status"] == "completed"
    assert data["result"]["num_documents"] == 3
    assert data["result"]["num_topics_used"] == 2
    assert len(data["result"]["rankings"]) == 3
    # Top match should be Doc 3
    assert data["result"]["rankings"][0]["doc_id"] == "Doc 3"
    assert len(data["result"]["topics"]) == 2
    assert len(data["result"]["documents"]) == 3






