# Crawl: Scratch-built HTML Parser & Web Crawling Engine

`crawl` is a production-quality, dependency-free Python library built entirely from scratch. It implements its own lexical tokenizer, tree construction parser, Document Object Model (DOM), CSS selector search engine, structured data extractor, HTML cleaner, and multi-format exporters.

This engine does not wrap third-party libraries like BeautifulSoup, lxml, Scrapy, Selenium, or Playwright. Instead, it serves as a pure demonstration of browser-like parsing and scraping internals using Python's standard library.

---

## Table of Contents
1. [Installation](#installation)
2. [CLI & Web UI User Interface](#cli--web-ui-user-interface)
3. [Technical Architecture](#technical-architecture)
4. [API Reference](#api-reference)
   - [Crawl](#crawl)
   - [Page](#page)
   - [Node](#node)
   - [Response](#response)
   - [Exceptions](#exceptions)
5. [Usage Recipes](#usage-recipes)
   - [Recipe 1: Local HTML Parsing & CSS Querying](#recipe-1-local-html-parsing--css-querying)
   - [Recipe 2: Live Crawling with Redirect History](#recipe-2-live-crawling-with-redirect-history)
   - [Recipe 3: Extracting Tables & Forms](#recipe-3-extracting-tables--forms)
   - [Recipe 4: Decoupled Parsing, Cleaning, & Exporting](#recipe-4-decoupled-parsing-cleaning--exporting)
6. [Release Notes & Versioning](#release-notes--versioning)

---

## Installation

The package is published on PyPI under the official distribution name **`crawl-jobs`**.

### 1. Install from PyPI

```bash
# Core package (crawler, DOM parser, extractor, exporters, ATS base, and NLP base)
pip install crawl-jobs
```

#### Optional Feature Extras
Crawl offers modular installation extras so you only install what you need:

```bash
# Web Dashboard & REST API (Flask UI + /api/nlp/* endpoints)
pip install "crawl-jobs[web]"

# Dense Neural Embeddings (sentence-transformers for ATS semantic similarity & NLP embeddings)
pip install "crawl-jobs[semantic]"

# Accelerated Matrix Analytics (NumPy for high-throughput SVD in LSI)
pip install "crawl-jobs[analytics]"

# Full Suite (installs all optional dependencies)
pip install "crawl-jobs[all]"
```

### 2. Optional Model & Corpus Data Setup
Crawl's NLP Lab and ATS components work with pure-Python fallbacks out of the box, but you can enable full linguistic models by downloading the corresponding resources:

- **WordNet Lemmatizer Corpora (NLTK)**:
  Required for NLTK WordNet morphological lemmatization in NLP Lab and ATS Stage 3:
  ```bash
  python -m nltk.downloader wordnet omw-1.4
  ```
- **spaCy English Model Pipeline**:
  Required for spaCy-based POS tagging and lemmatization:
  ```bash
  python -m spacy download en_core_web_sm
  ```

### 3. Developer Editable Mode (Recommended)
From the root directory of this repository:
```bash
pip install -e ".[all]"
```

### 4. Standard Local Installation
```bash
pip install .
```

### 5. Direct GitHub Installation
```bash
pip install git+https://github.com/vrushabh-hirap/crawl.git
```

---

## CLI & Web UI User Interface

Crawl includes a complete local CLI and web dashboard for executing distributed/multi-platform job searches.

### Running the Web App Interface
Start the local Flask server (which opens a beautifully styled neobrutalist browser dashboard at http://127.0.0.1:8080):
```bash
crawl webpage
```
To run on a specific port:
```bash
crawl webpage -p 9090
```

### Running via command-line
Execute searches and get structured lists directly in your terminal:
```bash
crawl search --title "Software Engineer" --location "Bangalore" --mode remote
```

### NLP Lab (Text Analytics Workbench)
The web dashboard includes an interactive **NLP Lab** section for running text analytics, linguistic normalization, language modeling, and semantic indexing:

1. **Dashboard UI**: Navigate to the **🧪 NLP Lab** tab at `http://127.0.0.1:8080`.
2. **Supported Tools**:
   - **Text Analysis**: Language detection, word count, sentence count, and token stream breakdown.
   - **Stemming & Lemmatization**: WordNet and spaCy lemmatization, Porter and Snowball stemmers.
   - **POS, Embeddings & Classification**: Part-of-speech syntax tagging, dense vector embeddings, and zero-shot categorization.
   - **TF-IDF Analysis**: Term frequency-inverse document frequency weighting, n-gram bounds, and key vocabulary scoring.
   - **N-gram Sequence Probability**: Statistical n-gram language models, sequence likelihood, and perplexity scoring.
   - **LSI Document Search & Categorization**: Latent Semantic Indexing (Truncated SVD) for semantic retrieval and topic clustering.
3. **REST API**:
   - `GET /api/nlp/tools`: Returns the list of registered tools, parameters, and sample texts.
   - `POST /api/nlp/run`: Validates and runs tools.
     ```bash
     curl -X POST http://127.0.0.1:8080/api/nlp/run \
       -H "Content-Type: application/json" \
       -d '{"tool_id": "stem_lemma", "text": "Engineers developing applications", "options": {"method": "wordnet"}}'
     ```
4. **Python API**:
   ```python
   from crawl.nlp import analyze_text, execute_nlp_tool

   # Run Text Analysis directly
   analysis = analyze_text(
       "Natural Language Processing enables computers to understand human language. It is fascinating!",
       options={"lowercase": False, "strip_punctuation": True}
   )

   print(analysis["language"])            # 'en'
   print(analysis["language_name"])       # 'English'
   print(analysis["is_uncertain"])        # False
   print(analysis["word_count"])          # 14
   print(analysis["sentence_count"])      # 2
   print(analysis["tokens"])              # ['Natural', 'Language', 'Processing', ...]

   # Or execute via the unified service dispatcher:
   payload = execute_nlp_tool("text_analysis", "Hello world", {"lowercase": True})
   ```

5. **Stemming & Lemmatization (WordNet, spaCy, Porter)**:
   Crawl provides morphological normalization through NLTK WordNet, spaCy `en_core_web_sm`, and NLTK Porter/Snowball stemmers.

   - **WordNet Data Setup**:
     WordNet lemmatization requires NLTK's `wordnet` and `omw-1.4` corpora:
     ```bash
     python -m nltk.downloader wordnet omw-1.4
     ```
     Or in Python:
     ```python
     import nltk
     nltk.download('wordnet')
     nltk.download('omw-1.4')
     ```
     *Note*: If WordNet data is missing, Crawl provides explicit instructions and will **never silently claim WordNet was used** if it fell back to another algorithm.

   - **Python API**:
     ```python
     from crawl.nlp import stem_and_lemmatize
     from crawl.jobs.ats.pipeline.stage3_normalizer import normalize, NormalizerConfig

     # 1. Run WordNet Lemmatization via NLP Lab API
     result = stem_and_lemmatize(
         "Engineers developing distributed web applications across universities.",
         options={"method": "wordnet", "pos": "all"}
     )
     print(result["method_used"])   # 'wordnet' (or 'spacy (fallback: wordnet missing)' if uninstalled)
     print(result["tokens"])        # ['engineer', 'develop', 'distribute', 'web', 'application', 'across', 'university']
     print(result["stats"])         # {'total_tokens': 7, 'transformed_tokens': 5, ...}

     # 2. Use WordNet in Crawl's ATS Résumé Normalizer (Stage 3)
     config = NormalizerConfig(method="wordnet", wordnet_pos="all")
     output = normalize("Senior engineer developing microservices", config)
     print(output.method_used)      # 'wordnet'
     print(output.tokens)           # ['senior', 'engineer', 'develop', 'microservices']
     ```

6. **POS, Embeddings & Classification**:
   Provides syntax parsing, vector embeddings, and text classification.

   - **Part-of-Speech Tagging**:
     Extracts fine-grained tags (Penn Treebank), coarse Universal POS, grammatical descriptions, and lexical density:
     ```python
     from crawl.nlp import pos_tag_text

     pos_info = pos_tag_text("The senior engineer deployed microservices.")
     print(pos_info["tagged_tokens"][0])
     # {'token': 'The', 'pos': 'DET', 'tag': 'DT', 'description': 'determiner', ...}
     print(pos_info["pos_distribution"])  # {'NOUN': 2, 'ADJ': 1, 'VERB': 1, ...}
     ```

   - **Text Embeddings**:
     Generates dense vector embeddings. Reuses the project's optional `sentence-transformers` dependency (`all-MiniLM-L6-v2`, 384 dimensions) when installed, or falls back to spaCy's built-in 96-dimensional vector space with actionable instructions:
     ```python
     from crawl.nlp import get_text_embedding

     emb = get_text_embedding("Senior software engineer building microservices.")
     print(emb["dimensions"])       # 384 (if sentence-transformers installed) or 96 (spaCy)
     print(emb["vector_preview"])   # [-0.0288, -0.1992, 0.5355, ...]
     print(emb["vector_norm"])      # 1.0 (unit normalized) or L2 norm
     ```
     *Optional Extra*: To enable 384-dimensional dense neural embeddings:
     ```bash
     pip install crawl-jobs[semantic]
     # or: pip install sentence-transformers
     ```

   - **Text Classification**:
     Classifies text across 5 benchmark categories (`Technology`, `Engineering`, `Science`, `Business`, `General`) using a calibrated prototype / Naive Bayes model with 100% benchmark accuracy, or accepts user-provided datasets:
     ```python
     from crawl.nlp import classify_text

     # 1. Classify using documented benchmark dataset
     result = classify_text("Senior backend engineer deploying Docker containers on AWS.")
     print(result["predicted_label"])          # 'Technology'
     print(result["confidence_percentage"])    # '99.2%'
     print(result["explanation"])              # Feature evidence & keyword matches

     # 2. Custom user-provided dataset flow
     custom_dataset = {
         "Aerospace": ["Orbital satellite propulsion trajectory aerodynamic lift."],
         "Biochemistry": ["Enzyme kinetics cellular DNA synthesis mitochondrial ATP."],
     }
     custom_res = classify_text("Satellite orbital trajectory.", dataset=custom_dataset)
     print(custom_res["predicted_label"])      # 'Aerospace'
     ```

7. **TF-IDF Multi-Document Analysis**:
   Computes Term Frequency - Inverse Document Frequency scores across multi-document collections to evaluate keyword significance and extract terms with high discriminatory power.

   - **Lightweight & Self-Contained**: Implemented in pure Python with smooth IDF and L2 vector normalization. Requires zero external dependencies. Completely decoupled from the ATS taxonomy matcher and sentence-transformer embeddings.
   - **Multi-Document Scoring**: Accepts multiple documents (split by double newlines `\n\n` or provided as a list).
   - **Distinguishing Terms Discovery**: Automatically calculates which keywords are unique or dominant in each document relative to the rest of the corpus.

   ```python
   from crawl.nlp import compute_tfidf

   documents = [
       "Python backend microservices using Flask and Redis database caching.",
       "Python machine learning pipelines using PyTorch neural networks.",
       "Kubernetes container orchestration with Docker on AWS cloud clusters.",
   ]

   tfidf = compute_tfidf(documents, options={"ngram_range": "1-1", "max_features": 10})

   print(tfidf["vocabulary"])  # ['python', 'using', 'kubernetes', 'container', ...]
   for doc in tfidf["documents"]:
       print(doc["doc_id"], "Distinctive Terms:", [d["term"] for d in doc["distinctive_terms"]])

   # Distinguishing analysis explanation:
   for insight in tfidf["distinguishing_analysis"]:
       print("•", insight)
   ```

8. **N-gram Language Modeling & Sequence Probability**:
   Crawl provides a self-contained statistical language model engine supporting configurable Unigrams ($n=1$), Bigrams ($n=2$), and Trigrams ($n=3$).

   - **Mathematical Foundations**:
     - **Chain Rule with Markov Assumption**:
       $$P(w_1, w_2, \dots, w_m) = \prod_{i=1}^m P(w_i \mid w_{i-n+1}^{i-1})$$
     - **Perplexity Scoring** (effective branching factor):
       $$PP(W) = P(W)^{-\frac{1}{m}} = 2^{-\frac{1}{m}\sum_{i=1}^m \log_2 P(w_i \mid \text{context})}$$

   - **Handling Unseen Sequences & Smoothing**:
     - **The Zero-Probability Problem**: Under standard Maximum Likelihood Estimation (MLE, `smoothing="none"`), if any n-gram transition has count $C = 0$, multiplying by zero causes the entire sequence likelihood to collapse to $0.0$, log probability to $-\infty$, and perplexity to $\infty$.
     - **Laplace (+1) Smoothing (`smoothing="laplace"`)**:
       $$P_{\text{Laplace}}(w_i \mid \text{context}) = \frac{C(w_{i-n+1}^i) + 1}{C(w_{i-n+1}^{i-1}) + V}$$
       where $V = |\mathcal{V}|$ is the vocabulary size. Even when an n-gram is unseen ($C = 0$), the numerator becomes $1$ and the denominator $C_{\text{context}} + V > 0$, guaranteeing a non-zero probability and finite perplexity.
     - **Add-$k$ Smoothing (`smoothing="add_k"` with configurable $k$)**:
       $$P_{\text{Add-}k}(w_i \mid \text{context}) = \frac{C(w_{i-n+1}^i) + k}{C(w_{i-n+1}^{i-1}) + k \cdot V}$$

   - **Python API**:
     ```python
     from crawl.nlp import calculate_sequence_probability, NGramLanguageModel

     corpus = "to be or not to be that is the question whether tis nobler in the mind"

     # Calculate sequence probability with Bigrams (n=2) and Laplace smoothing:
     result = calculate_sequence_probability(
         corpus=corpus,
         target_sequence="to be or not to be",
         n=2,
         smoothing="laplace",
     )

     print("Order:", result["order_name"])                   # 'Bigram'
     print("Joint Probability:", result["total_probability_scientific"]) # '1.2658e-07'
     print("Perplexity:", result["perplexity"])             # 14.1125
     print("Unseen Transitions:", result["unseen_ngram_count"]) # 0

     # Inspect explainable transition steps for assignments:
     for step in result["steps"]:
         print(f"Step {step['step']}: {step['formula_str']} -> p={step['probability_formatted']}")
     ```

9. **Latent Semantic Indexing (LSI / Truncated SVD)**:
   Crawl provides a self-contained Latent Semantic Indexing (LSI) engine for document collections, applying Truncated Singular Value Decomposition (SVD) on TF-IDF document-term matrices to discover hidden concept correlations.

   - **Mathematical Principles**:
     - **Truncated SVD**: Decomposes the $N \times M$ TF-IDF matrix into $X \approx U_k \Sigma_k V_k^T$, where $k$ represents the number of latent semantic dimensions.
     - **Document Projections**: Documents are embedded into $k$-dimensional latent coordinates $\mathbf{d}_i = U_k \Sigma_k$.
     - **Semantic Query Retrieval**: A query vector $\mathbf{q}$ is projected into the latent space via $\mathbf{q}_k = \mathbf{q} V_k$ and ranked by cosine similarity against document coordinates:
       $$\text{CosineSim}(\mathbf{q}_k, \mathbf{d}_i) = \frac{\mathbf{d}_i \cdot \mathbf{q}_k}{\|\mathbf{d}_i\|_2 \|\mathbf{q}_k\|_2}$$
     - **Synonymy Resolution**: Documents sharing latent topics are retrieved even when exact query terms differ lexically.
     - **Safe Small-Collection Bounds**: The component count $k$ is automatically validated and clamped to $k_{\text{eff}} = \max(1, \min(k, N, M))$, guaranteeing that small inputs never cause matrix dimension errors.

   - **Architectural Distinction**:
     LSI is a classic linear algebraic matrix factorization technique. It is completely independent of the ATS keyword taxonomy and the optional deep neural `sentence-transformers` feature (`all-MiniLM-L6-v2`).

   - **Installation & Dependencies**:
     LSI uses `numpy.linalg.svd` when `numpy` is installed (optional extra: `pip install crawl-jobs[analytics]`), and provides an automatic self-contained pure-Python SVD fallback if numpy is absent.

   - **Python API**:
     ```python
     from crawl.nlp import LatentSemanticIndexing, run_lsi_analysis

     documents = [
         "Python web development, Flask REST APIs, and PostgreSQL database queries.",
         "Scalable Python backend engineering, Docker microservices architecture, and Kubernetes.",
         "Deep learning neural networks, computer vision, and transformer models in PyTorch.",
         "Machine learning model training, transformers, and deep neural networks in TensorFlow.",
     ]

     # 1. Semantic Search in Latent Space
     results = run_lsi_analysis(
         documents=documents,
         query="scalable backend microservices docker",
         num_topics=2,
         mode="search",
         show_coordinates=True,
     )

     for rank in results["rankings"]:
         print(f"#{rank['rank']} {rank['doc_id']}: Cosine={rank['similarity_percentage']} ({rank['snippet']})")

     # 2. Topic Components & Singular Values
     for topic in results["topics"]:
         print(f"{topic['topic_id']} (σ={topic['singular_value']}): {topic['summary']}")

     # 3. Document Categorization / Clustering
     cat_results = run_lsi_analysis(documents, num_topics=2, mode="categorize")
     for cat in cat_results["categories"]:
         print(f"{cat['topic_id']}: {cat['assigned_documents_count']} documents")
     ```

10. **Runnable Examples**:
   - Text Analysis demonstration:
     ```bash
     python examples/nlp_text_analysis_demo.py
     ```
   - WordNet Lemmatizer & morphology comparison:
     ```bash
     python examples/nlp_wordnet_lemmatizer_demo.py
     ```
   - POS, Embeddings & Classification demonstration:
     ```bash
     python examples/nlp_pos_embeddings_classification_demo.py
     ```
   - TF-IDF multi-document analysis:
     ```bash
     python examples/nlp_tfidf_demo.py
     ```
   - N-gram Language Model & Smoothing demonstration:
     ```bash
     python examples/nlp_ngram_demo.py
     ```
   - Latent Semantic Indexing (LSI) search & categorization:
     ```bash
     python examples/nlp_lsi_demo.py
     ```

---

## Technical Architecture

The library is decoupled into distinct, modular subsystems:

```
                          ┌──────────────┐
                          │  Downloader  │
                          └──────┬───────┘
                                 │ HTTP Response
                                 ▼
                          ┌──────────────┐
                          │  Tokenizer   │
                          └──────┬───────┘
                                 │ Token Stream
                                 ▼
                          ┌──────────────┐
                          │    Parser    │
                          └──────┬───────┘
                                 │ DOM Tree
                                 ▼
 ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
 │   Selector   │◄───────►│   DOM Node   │◄───────►│  Extractor   │
 └──────────────┘         └──────┬───────┘         └──────────────┘
                                 │ In-Place Cleanup
                                 ▼
                          ┌──────────────┐         ┌──────────────┐
                          │   Cleaner    ├────────►│   Exporter   │
                          └──────────────┘         └──────────────┘
```

1. **Downloader (`downloader.py`)**: Uses Python's built-in `urllib.request` HTTPS handlers to negotiate request headers, enforce timeouts, handle redirects, capture redirect hop chains, and configure SSL/TLS verification contexts.
2. **Lexical Tokenizer (`tokenizer.py`)**: A character-by-character scanner implemented as a state machine. It converts raw markup text into discrete tags (`START_TAG`, `END_TAG`), character arrays (`TEXT`), docstrings (`DOCTYPE`), and comment blocks (`COMMENT`).
3. **DOM Parser (`parser.py`)**: A stack-based builder that processes the token stream. It auto-closes unclosed nested tags based on parent tags and skips standard HTML void elements (such as `img`, `br`, `hr`, `input`, `meta`, `link`).
4. **DOM Tree (`dom.py`)**: Represents elements, text blocks, comment lines, and doctype markers as `Node` structures. Supports DOM tree navigation (`append_child`, `remove_child`) and serialization (`inner_html`, `outer_html`).
5. **Selector Engine (`selector.py`)**: Evaluates CSS queries left-to-right, processing tags, IDs, class names, descendant spacing, direct children (`>`), and comma lists, keeping nodes sorted in document pre-order.
6. **Data Extractor (`extractor.py`)**: Isolates headers, paragraphs, titles, tables, forms, list items, links, and buttons, automatically resolving relative paths.
7. **DOM Cleaner (`cleaner.py`)**: Collapses multiple whitespaces, decodes HTML entities, prunes empty nodes (ignoring void tags), and removes identical sibling subtrees.
8. **Exporter (`exporter.py`)**: Outputs raw data arrays to structured CSV, JSON, or TXT formats.

---

## API Reference

### `Crawl`
The high-level user interface responsible for executing network transfers and returning parsed results.

* **Constructor**:
  ```python
  from crawl import Crawl
  crawler = Crawl(
      default_headers: dict = None,   # Merged with standard request headers
      timeout: float = 10.0,          # Connection timeout in seconds
      verify_ssl: bool = True,         # Toggles SSL certificate verification
      rate_limit_delay: float = 0.0,  # Delay (in seconds) between requests
      obey_robots: bool = False       # Respect robots.txt directives (extension placeholder)
  )
  ```
* **Methods**:
  * **`fetch(url: str, headers: dict = None, timeout: float = None) -> Page`**: Downloads the target URL, parses its content, and returns a `Page` object.

---

### `Page`
Returned by `Crawl.fetch()`. It represents a downloaded, parsed HTML document.

* **Properties**:
  * **`url`**: The final absolute URL of the page (after redirects) *(type: `str`)*.
  * **`response`**: The raw network response object *(type: `Response`)*.
  * **`node`**: The document's root Node *(type: `Node`)*.
* **Extraction Methods**:
  * **`title() -> str`**: Gets the page title (falls back to the first `<h1>` text if missing).
  * **`paragraphs() -> List[str]`**: Gets the text of all `<p>` paragraph tags.
  * **`images() -> List[str]`**: Gets absolute URLs of all `<img>` src paths.
  * **`links() -> List[str]`**: Gets absolute URLs of all `<a>` anchor href paths.
  * **`tables() -> List[List[List[str]]]`**: Gets row/column contents for all tables on the page.
  * **`forms() -> List[Dict[str, Any]]`**: Gets metadata for all form actions, methods, and inputs.
  * **`lists() -> List[List[str]]`**: Gets list items (`<li>`) grouped by parent list (`<ul>`/`<ol>`).
  * **`buttons() -> List[Dict[str, str]]`**: Gets button nodes and button-type inputs (submit, reset).
  * **`meta_tags() -> List[Dict[str, str]]`**: Gets meta name, property, and content descriptors.
  * **`headers() -> List[Dict[str, str]]`**: Gets headings (H1–H6) in document order.
* **DOM & Traversal Methods**:
  * **`find(selector: str) -> Optional[Node]`**: Returns the first node matching the CSS selector.
  * **`find_all(selector: str) -> List[Node]`**: Returns all nodes matching the CSS selector.
  * **`get_text() -> str`**: Returns raw concatenated text content of the entire document.
  * **`inner_html() -> str`**: Returns the serialized inner HTML of the document's children.
  * **`outer_html() -> str`**: Returns the serialized outer HTML of the full DOM tree.
* **Cleaning & Export Methods**:
  * **`clean() -> None`**: Performs full DOM whitespace normalization, entity decoding, and duplicate sibling pruning in-place.
  * **`export_json(filepath: str) -> None`**: Exports basic page metadata to a local JSON file.

---

### `Node`
A class representing a single element, text node, comment, or doctype.

* **Properties**:
  * **`tag`**: Tag name in lowercase (e.g. `'div'`), or `None` for text nodes, or `#comment`/`#doctype` *(type: `Optional[str]`)*.
  * **`attributes`**: Attribute key-value mapping (lowercased keys) *(type: `dict`)*.
  * **`children`**: List of child DOM nodes *(type: `List[Node]`)*.
  * **`parent`**: Parent element node, or `None` if it is a root *(type: `Optional[Node]`)*.
  * **`text`**: Raw text content (used for text, comment, and doctype nodes) *(type: `Optional[str]`)*.
* **Methods**:
  * **`append_child(child: Node) -> None`**: Appends a child Node, managing parent links.
  * **`remove_child(child: Node) -> None`**: Removes a child Node.
  * **`get_attribute(name: str, default: str = None) -> Optional[str]`**: Case-insensitive attribute lookup.
  * **`set_attribute(name: str, value: str) -> None`**: Sets or overrides a custom attribute.
  * **`find(selector: str) -> Optional[Node]`**: Performs CSS selector lookup starting from this node.
  * **`find_all(selector: str) -> List[Node]`**: Performs descendant CSS selector lookup.
  * **`get_text() -> str`**: Returns recursive text inside this Node.
  * **`inner_html() -> str`**: Serializes HTML contents of this Node's children.
  * **`outer_html() -> str`**: Serializes the HTML structure of this node and its children.

---

### `Response`
Represents the result of an HTTP transaction.

* **Properties**:
  * **`url`**: Final landing URL (after resolving redirects) *(type: `str`)*.
  * **`status_code`**: HTTP status response code (e.g., `200`, `404`) *(type: `int`)*.
  * **`headers`**: Dictionary of case-insensitive response headers *(type: `dict`)*.
  * **`content`**: Raw binary content *(type: `bytes`)*.
  * **`text`**: Decoded content string (automatically parses charset from headers) *(type: `str`)*.
  * **`redirect_history`**: List of redirect hops: `[{'status': int, 'url': str, 'headers': dict}]` *(type: `List[dict]`)*.

---

### Exceptions

* **`CrawlError`**: Base exception class.
* **`DownloadError`**: Connection issues, timeouts, or SSL failures.
* **`ParserError`**: Unrecoverable syntax errors during tokenization or parsing.
* **`InvalidSelector`**: Invalid CSS selector structure.
* **`NodeNotFound`**: Missing node during critical DOM traversals.
* **`ExportError`**: File writing or serialization constraints failures.

---

## Usage Recipes

### Recipe 1: Local HTML Parsing & CSS Querying
You can parse a string of HTML and query elements directly using CSS selectors.

```python
from crawl.parser import parse_html

html_content = """
<div id="container">
    <h1 class="title">Product Catalog</h1>
    <ul class="items">
        <li class="item" id="p-1">Laptop <span class="badge">Sale</span></li>
        <li class="item" id="p-2">Phone</li>
    </ul>
</div>
"""

# 1. Parse string to DOM Node
root = parse_html(html_content)

# 2. Query by ID
container = root.find("#container")
print("Container ID attribute:", container.get_attribute("id"))

# 3. Query by Class
title_node = root.find(".title")
print("Title Text:", title_node.get_text())

# 4. Direct Child Selector (ul > li)
list_items = root.find_all("ul.items > li")
for item in list_items:
    print(f"Item text: {item.get_text()} | ID: {item.get_attribute('id')}")

# 5. Descendant Selector (div span)
badges = root.find_all("div .badge")
for badge in badges:
    print("Found badge:", badge.get_text())
```

---

### Recipe 2: Live Crawling with Redirect History
Fetch a URL and inspect its headers, status, and redirect hops.

```python
from crawl import Crawl, CrawlError

crawler = Crawl(timeout=5.0, verify_ssl=True)

try:
    # URL redirects http -> https
    page = crawler.fetch("http://github.com")
    
    print("Final Landing URL:", page.url)
    print("Response Status Code:", page.response.status_code)
    print("Server Header:", page.response.headers.get("server"))
    
    # Inspect redirect history
    if page.response.redirect_history:
        print("\nRedirect Chain:")
        for idx, hop in enumerate(page.response.redirect_history):
            print(f" Hop {idx + 1}: Code {hop['status']} -> {hop['url']}")

except CrawlError as e:
    print("Crawling failed:", e)
```

---

### Recipe 3: Extracting Tables & Forms
Extract structured metadata from tables and interactive form fields.

```python
from crawl.parser import parse_html
from crawl.extractor import extract_tables, extract_forms

html_content = """
<div>
    <table>
        <tr><th>Processor</th><th>Cores</th></tr>
        <tr><td>Ryzen 9</td><td>16</td></tr>
        <tr><td>Core i9</td><td>24</td></tr>
    </table>

    <form action="/login" method="POST">
        <input type="text" name="username" value="user1">
        <input type="password" name="password">
        <button type="submit">Log In</button>
    </form>
</div>
"""

root = parse_html(html_content)

# Extract Tables
tables = extract_tables(root)
print("Parsed Table Core List:")
for table in tables:
    for row in table:
        print(f" - {row[0]} has {row[1]} cores")

# Extract Forms
forms = extract_forms(root, base_url="https://mysite.com")
for form in forms:
    print(f"\nForm URL: {form['action']} | Method: {form['method']}")
    for field in form["inputs"]:
        print(f"  Field: tag={field['tag']}, name={field['name']}, default={field.get('value')}")
```

---

### Recipe 4: Decoupled Parsing, Cleaning, & Exporting
Decouple parsing from cleaning and output the results to JSON, CSV, or TXT.

```python
import os
from crawl.parser import parse_html
from crawl.cleaner import clean_dom
from crawl.extractor import extract_links
from crawl.exporter import export_json, export_csv, export_txt

html_content = """
<html>
<body>
    <h1>   Clean   Me   &amp;   Export   </h1>
    <p>   Some paragraph  </p>
    <a href="/doc1">Doc 1</a>
    <a href="/doc2">Doc 2</a>
    <div class="empty-garbage"></div> <!-- gets pruned -->
</body>
</html>
"""

# 1. Parse DOM
root = parse_html(html_content)

# 2. Clean tree (normalizes text whitespace and entity unescaping)
clean_dom(root)

# 3. Extract Links
links = extract_links(root, base_url="https://site.com")

# 4. Export
os.makedirs("./exports", exist_ok=True)
export_json(links, "./exports/links.json")
export_csv(links, "./exports/links.csv")
export_txt(root.get_text(), "./exports/page_text.txt")

print("Files saved in './exports/' successfully!")
```

---

## Release Notes & Versioning

### Version 1.2.0 (Current Release)
**Release Highlights:**
- **NLP Lab Workbench**: Added an interactive web interface tab at `/` (under the **🧪 NLP Lab** tab) and REST API endpoints (`GET /api/nlp/tools`, `POST /api/nlp/run`) providing 6 text analytics tools.
- **Text Analysis**: Detects natural language (with uncertainty signaling), sentence count, word count, character count, and cleaned token streams.
- **Stemming & Lemmatization**: Added WordNet morphological lemmatizer with explicit POS tag filtering and missing-data instructions, alongside spaCy lemmatization and Porter/Snowball stemmers.
- **POS, Embeddings & Classification**: Part-of-Speech syntax tagging with Penn Treebank & Universal POS annotations, dense vector embeddings with optional `sentence-transformers` integration, and calibrated text classification across documented benchmarks or user-provided datasets.
- **TF-IDF Multi-Document Analysis**: Computes document-term matrices, n-gram ranges, stopword pruning, and distinctive term discovery with pure-Python fallback.
- **N-gram Language Modeling**: Statistical Unigram, Bigram, and Trigram language modeling with Laplace and add-$k$ smoothing to prevent zero-probability collapse on unseen sequences, along with joint probability and perplexity calculations.
- **Latent Semantic Indexing (LSI)**: Truncated SVD decomposition of TF-IDF matrices for latent semantic search, topic component loadings, document coordinate inspection, and clustering. Automatically clamps $k \le \min(N, M)$ to guarantee zero-crash execution on small inputs.
- **Public Package Interfaces**: Cleanly exposed all algorithms under `crawl.nlp` and added 6 new runnable demonstration scripts in `examples/`.
- **Distribution & Extras Alignment**: Aligned PyPI package distribution name to `crawl-jobs`, bumped version to `1.2.0`, added `analytics` (`numpy`) and `all` extras, and updated documentation.
- **Full Backward Compatibility**: All 104 tests pass across the HTML tokenizer, DOM parser, crawler engine, job search adapters, and 7-stage ATS resume pipeline.

### Version 1.1.0
- Added 7-stage ATS Résumé Checker pipeline (`crawl.jobs.ats`) with document ingestion (PDF, DOCX, TXT), section segmentation, keyword matching, entity extraction, and reporting.

### Version 1.0.0
- Added Job Search engine and neobrutalist web dashboard (`crawl webpage`).

### Version 0.1.2
- Initial release featuring scratch-built HTML tokenizer, DOM tree, CSS selector engine, cleaner, and exporters.

