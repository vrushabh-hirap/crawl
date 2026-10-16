"""Example demonstrating TF-IDF analysis in Crawl's NLP Lab.

This runnable example shows:
1. Calculating Term Frequency - Inverse Document Frequency across multi-document collections.
2. Generating the corpus vocabulary and document-term weight matrix.
3. How TF-IDF weights identify distinctive terms that differentiate documents from one another.
4. Using n-gram features (unigrams and bigrams).
5. Executing TF-IDF analysis via the unified NLP Lab service dispatcher.
"""

from crawl.nlp import compute_tfidf, execute_nlp_tool


def main() -> None:
    print("=" * 78)
    print("CRAWL NLP LAB — TF-IDF MULTI-DOCUMENT ANALYSIS DEMONSTRATION")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # Scenario 1: Multi-Document Corpus & Distinctive Term Identification
    # -------------------------------------------------------------------------
    print("\n--- Scenario 1: Multi-Document Analysis & Term Discrimination ---")
    documents = [
        # Document 1: Web Development
        "Python backend microservices using Flask framework, REST APIs, and Redis database caching.",
        # Document 2: Machine Learning
        "Python machine learning pipelines using PyTorch neural networks, transformers, and model evaluation.",
        # Document 3: Cloud DevOps
        "Kubernetes container orchestration with Docker on AWS cloud infrastructure clusters.",
    ]

    print("Input Documents:")
    for i, doc in enumerate(documents, 1):
        print(f"  [Doc {i}]: {doc}")

    # Compute TF-IDF
    tfidf_results = compute_tfidf(documents, {"ngram_range": "1-1", "max_features": 12})

    print(f"\nCorpus Summary:")
    print(f"  Total Documents : {tfidf_results['num_documents']}")
    print(f"  Vocabulary Size : {tfidf_results['vocabulary_size']} unique terms")
    print(f"  Top Features    : {', '.join(tfidf_results['vocabulary'])}")

    # -------------------------------------------------------------------------
    # Scenario 2: How TF-IDF Distinguishes Documents
    # -------------------------------------------------------------------------
    print("\n--- Scenario 2: Document-Specific Discriminatory Terms ---")
    print("Notice how 'python' appears in both Doc 1 and Doc 2, giving it lower IDF (1.29),")
    print("while terms unique to single documents ('caching', 'pytorch', 'kubernetes') receive higher weights (1.69).\n")

    for doc_info in tfidf_results["documents"]:
        print(f"► {doc_info['doc_id']} ({doc_info['snippet']}):")
        print("  Top Weighted Terms:")
        for t in doc_info["top_terms"][:4]:
            print(f"    • {t['term']:14} : TF-IDF={t['score']:.4f} (TF={t['tf']}, IDF={t['idf']:.4f})")

        distinctive = doc_info["distinctive_terms"]
        if distinctive:
            dist_str = ", ".join(f"{d['term']} (+{d['uniqueness_margin']:.3f})" for d in distinctive[:3])
            print(f"  Distinguishing Features : {dist_str}")
        print()

    # -------------------------------------------------------------------------
    # Scenario 3: Document-Term Weight Matrix Table
    # -------------------------------------------------------------------------
    print("--- Scenario 3: Document-Term Matrix ---")
    print(f"{'Term':16} | {'DF':4} | {'IDF':7} | {'Doc 1':7} | {'Doc 2':7} | {'Doc 3':7}")
    print("-" * 62)
    for row in tfidf_results["matrix"][:10]:
        scores = row["doc_scores"]
        s1 = f"{scores[0]:.4f}" if len(scores) > 0 and scores[0] > 0 else "   -   "
        s2 = f"{scores[1]:.4f}" if len(scores) > 1 and scores[1] > 0 else "   -   "
        s3 = f"{scores[2]:.4f}" if len(scores) > 2 and scores[2] > 0 else "   -   "
        print(f"{row['term']:16} | {row['df']:<4} | {row['idf']:<7.4f} | {s1:7} | {s2:7} | {s3:7}")

    # -------------------------------------------------------------------------
    # Scenario 4: N-Gram Combinations (Unigrams + Bigrams)
    # -------------------------------------------------------------------------
    print("\n--- Scenario 4: N-Gram Support (1-2 Range) ---")
    bigram_results = compute_tfidf(documents, {"ngram_range": "1-2", "max_features": 8})
    print("Top Features including compound phrases:")
    for row in bigram_results["matrix"][:8]:
        term_type = "Bigram" if " " in row["term"] else "Unigram"
        print(f"  • {row['term']:28} [{term_type:7}] Aggregate Score: {row['aggregate_score']:.4f}")

    # -------------------------------------------------------------------------
    # Scenario 5: Service Dispatcher Execution
    # -------------------------------------------------------------------------
    print("\n--- Scenario 5: Unified Service Dispatcher Execution ---")
    raw_text_block = (
        "Information retrieval systems leverage term frequency and inverse document frequency.\n\n"
        "Search engines index web pages into vector representations for keyword retrieval."
    )
    api_payload = execute_nlp_tool(
        tool_id="tfidf",
        text=raw_text_block,
        options={"ngram_range": "1-1", "max_features": 6},
    )
    print(f"Success    : {api_payload['success']}")
    print(f"Status     : {api_payload['status']}")
    print(f"Documents  : {api_payload['result']['num_documents']}")
    print(f"Top Terms  : {api_payload['result']['vocabulary']}")
    print(f"Insights   : {api_payload['result']['distinguishing_analysis']}")

    print("\n" + "=" * 78)
    print("TF-IDF DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 78)


if __name__ == "__main__":
    main()
