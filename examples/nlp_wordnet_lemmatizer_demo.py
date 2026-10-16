"""Example demonstrating WordNet lemmatization, spaCy lemmatization, and stemming in Crawl.

This runnable example shows:
1. Normalizing natural language text with WordNet lemmatization.
2. Side-by-side comparison of WordNet lemmatizer, spaCy lemmatizer, and Porter stemmer.
3. WordNet POS tag configuration (all, noun, verb, adjective, adverb).
4. Integrating WordNet lemmatization into Crawl's Stage 3 ATS text normalizer.
5. Verification of the `method_used` attribute (ensuring fallback methods are never masked).
6. Missing NLTK data handling and clear installation instructions.
7. Dispatching execution via the unified NLP Lab service layer.
"""

from crawl.nlp import stem_and_lemmatize, execute_nlp_tool
from crawl.jobs.ats.pipeline.stage3_normalizer import normalize, NormalizerConfig


def main() -> None:
    print("=" * 76)
    print("CRAWL NLP LAB — WORDNET LEMMATIZER & MORPHOLOGY DEMO")
    print("=" * 76)

    sample_sentence = (
        "The senior software engineers were developing distributed web applications, "
        "running automated test suites across universities, and achieving better performance."
    )

    # -------------------------------------------------------------------------
    # Scenario 1: WordNet Lemmatization as an Explicit Option
    # -------------------------------------------------------------------------
    print("\n--- Scenario 1: WordNet Lemmatization (Multi-POS Analysis) ---")
    wn_res = stem_and_lemmatize(sample_sentence, {"method": "wordnet", "pos": "all"})

    print(f"Method Requested : {wn_res['method_requested']}")
    print(f"Method Used      : {wn_res['method_used']} (Fallback: {wn_res['is_fallback']})")
    print(f"Algorithm        : {wn_res['algorithm']}")
    print(f"Tokens Count     : {wn_res['stats']['total_tokens']}")
    print(f"Modified Tokens  : {wn_res['stats']['transformed_tokens']} ({wn_res['stats']['transformation_rate'] * 100:.1f}%)")
    print("\nKey Lemmatized Words:")
    for item in wn_res["items"]:
        if item["changed"]:
            print(f"  • {item['original']:15} ➔ {item['normalized']:12} [POS: {item['pos']}]")

    # -------------------------------------------------------------------------
    # Scenario 2: Method Comparison (WordNet vs spaCy vs Porter)
    # -------------------------------------------------------------------------
    print("\n--- Scenario 2: Comparative Morphological Analysis ---")
    test_words = [
        "developing",
        "universities",
        "running",
        "better",
        "matrices",
        "engineering",
        "leaves",
    ]
    test_text = " ".join(test_words)

    res_wordnet = stem_and_lemmatize(test_text, {"method": "wordnet", "pos": "all"})
    res_spacy   = stem_and_lemmatize(test_text, {"method": "spacy"})
    res_porter  = stem_and_lemmatize(test_text, {"method": "porter"})

    print(f"{'Original':14} | {'WordNet (Method: ' + res_wordnet['method_used'] + ')':30} | {'spaCy (' + res_spacy['method_used'] + ')':20} | {'Porter (' + res_porter['method_used'] + ')'}")
    print("-" * 88)
    for w, w_tok, s_tok, p_tok in zip(test_words, res_wordnet["tokens"], res_spacy["tokens"], res_porter["tokens"]):
        print(f"{w:14} | {w_tok:30} | {s_tok:20} | {p_tok}")

    # -------------------------------------------------------------------------
    # Scenario 3: WordNet POS Tag Filtering
    # -------------------------------------------------------------------------
    print("\n--- Scenario 3: WordNet Part-of-Speech Tag Filtering ---")
    phrase = "The meeting was meeting all performance expectations."
    print(f"Phrase: {phrase!r}")

    res_all = stem_and_lemmatize(phrase, {"method": "wordnet", "pos": "all"})
    res_v   = stem_and_lemmatize(phrase, {"method": "wordnet", "pos": "v"})
    res_n   = stem_and_lemmatize(phrase, {"method": "wordnet", "pos": "n"})

    print(f"  POS='all' : {' '.join(res_all['tokens'])}")
    print(f"  POS='v'   : {' '.join(res_v['tokens'])}")
    print(f"  POS='n'   : {' '.join(res_n['tokens'])}")

    # -------------------------------------------------------------------------
    # Scenario 4: Crawl Stage 3 ATS Normalizer with WordNet
    # -------------------------------------------------------------------------
    print("\n--- Scenario 4: Stage 3 ATS Normalizer with WordNet ---")
    resume_snippet = "Experienced engineer developing CI/CD microservices in Python across top universities."
    config = NormalizerConfig(method="wordnet", wordnet_pos="all")
    normalizer_output = normalize(resume_snippet, config)

    print(f"Normalizer Output Method Used : {normalizer_output.method_used}")
    print(f"Normalizer Output Warning     : {normalizer_output.warning}")
    print(f"Normalized Tokens             : {normalizer_output.tokens[:8]}...")
    print(f"Acronym Expansion             : {normalizer_output.expanded_text}")

    # -------------------------------------------------------------------------
    # Scenario 5: Missing NLTK WordNet Data Handling & Fallback Transparency
    # -------------------------------------------------------------------------
    print("\n--- Scenario 5: Missing Resource Handling & Clear Instructions ---")
    import crawl.nlp.tools.stem_lemma as sl

    orig_loader = sl.get_wordnet_lemmatizer
    try:
        # Simulate uninstalled / missing WordNet data
        sl.get_wordnet_lemmatizer = lambda: None

        # 5a. Explicit error mode (raise_on_missing)
        print("[5a] Missing data with raise_on_missing:")
        try:
            sl.stem_and_lemmatize("Testing missing resource", {"method": "wordnet", "fallback_on_missing": False})
        except LookupError as e:
            print("  Caught Expected LookupError:")
            for line in str(e).splitlines():
                print(f"    {line}")

        # 5b. Fallback mode (transparently reports fallback method)
        print("\n[5b] Missing data with fallback enabled:")
        res_fb = sl.stem_and_lemmatize("Developing applications", {"method": "wordnet", "fallback_on_missing": True})
        print(f"  Method Requested : {res_fb['method_requested']}")
        print(f"  Method Used      : {res_fb['method_used']} (Fallback: {res_fb['is_fallback']})")
        print(f"  Warning Attached : {res_fb['warning'] is not None}")
        print("  Notice: WordNet was NOT silently claimed as the generator!")
    finally:
        sl.get_wordnet_lemmatizer = orig_loader

    # -------------------------------------------------------------------------
    # Scenario 6: Dispatching via Unified NLP Lab API
    # -------------------------------------------------------------------------
    print("\n--- Scenario 6: Execution via NLP Lab Service Layer ---")
    api_response = execute_nlp_tool(
        tool_id="stem_lemma",
        text="The software architects were deploying scalable microservices.",
        options={"method": "wordnet", "pos": "all"},
    )
    print(f"Success    : {api_response['success']}")
    print(f"Tool Name  : {api_response['tool_name']}")
    print(f"Status     : {api_response['status']}")
    print(f"Method Used: {api_response['result']['method_used']}")
    print(f"Tokens     : {api_response['result']['tokens']}")

    print("\n" + "=" * 76)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 76)


if __name__ == "__main__":
    main()
