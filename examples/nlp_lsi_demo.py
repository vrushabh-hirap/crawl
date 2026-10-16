"""Example demonstrating Latent Semantic Indexing (LSI) in Crawl's NLP Lab.

This runnable example demonstrates:
1. Building a TF-IDF document-term matrix over a multi-document collection.
2. Applying Truncated SVD to reduce vocabulary dimensions to latent semantic concepts.
3. Inspecting latent topics, singular values, variance explained, and term loadings.
4. Projecting documents into the latent space and displaying their coordinates.
5. Semantic query search: ranking documents by cosine similarity in the reduced latent space.
6. Document categorization: grouping documents by dominant latent topic component.
7. Handling boundary conditions: safe clamping when requested k exceeds collection rank.
8. Execution via Crawl's unified NLP Lab service dispatcher.
"""

from crawl.nlp import (
    LatentSemanticIndexing,
    run_lsi_analysis,
    execute_nlp_tool,
)


def main() -> None:
    print("=" * 80)
    print("CRAWL NLP LAB — LATENT SEMANTIC INDEXING (LSI / TRUNCATED SVD)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Sample Document Collection (2 Main Latent Clusters: Web/Backend vs. Deep Learning)
    # -------------------------------------------------------------------------
    documents = [
        # Cluster A: Web Development & Microservices
        "Python web development, Flask REST APIs, and PostgreSQL database queries.",
        # Cluster A: Cloud Backend & Microservices
        "Scalable Python backend engineering, Docker microservices architecture, and Kubernetes.",
        # Cluster B: Deep Learning
        "Deep learning neural networks, computer vision, and transformer models in PyTorch.",
        # Cluster B: Machine Learning Pipelines
        "Machine learning model training, transformers, and deep neural networks in TensorFlow.",
    ]

    print("\n--- Document Collection (N = 4) ---")
    for i, doc in enumerate(documents, 1):
        print(f"  [Doc {i}]: {doc}")

    # -------------------------------------------------------------------------
    # Scenario 1: Model Fitting & Latent Topic Extraction
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 1: Truncated SVD & Latent Topics (k = 2)")
    print("=" * 80)

    lsi = LatentSemanticIndexing(num_topics=2)
    lsi.fit(documents)

    print(f"Engine Used     : {lsi.engine}")
    print(f"Vocabulary Size : {len(lsi.vocabulary)} unique terms")
    print(f"Effective k     : {lsi.k_effective} latent topics")
    print(f"Singular Values : {lsi.singular_values}")
    print(f"Variance Ratios : {lsi.variance_explained_ratio}")

    print("\nLatent Topic Components (Top Vocabulary Terms):")
    topics = lsi.get_topics(num_terms=4)
    for t in topics:
        print(f"  ► {t['topic_id']} (σ = {t['singular_value']:.4f}, {t['variance_explained']} variance):")
        for term_info in t["top_terms"]:
            sign_str = "+" if term_info["weight"] >= 0 else ""
            print(f"      • {term_info['term']:16} : Loading = {sign_str}{term_info['weight']:.4f}")

    # -------------------------------------------------------------------------
    # Scenario 2: Document Coordinates in Latent Space
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 2: Document Projections (Latent Coordinates)")
    print("=" * 80)

    doc_coords = lsi.get_document_coordinates()
    for d in doc_coords:
        coords_str = ", ".join(f"t{idx+1}={val:+.4f}" for idx, val in enumerate(d["coordinates"]))
        print(f"  • {d['doc_id']:6} | Dominant: {d['dominant_topic']:8} | Coords: [{coords_str}]")
        print(f"           Text: {d['snippet']}")

    # -------------------------------------------------------------------------
    # Scenario 3: Semantic Query Search
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 3: Latent Semantic Search (Cosine Similarity)")
    print("=" * 80)

    query = "scalable backend microservices docker"
    print(f'Search Query: "{query}"\n')

    rankings = lsi.search(query)
    print(f"{'Rank':4} | {'Doc ID':7} | {'Cosine Sim':10} | {'Percentage':10} | {'Dominant Topic':14}")
    print("-" * 60)
    for r in rankings:
        print(f"{r['rank']:<4} | {r['doc_id']:7} | {r['similarity']:<10.4f} | {r['similarity_percentage']:<10} | {r['dominant_topic']:14}")
        print(f"       Snippet: {r['snippet']}")

    # -------------------------------------------------------------------------
    # Scenario 4: Topic Categorization / Clustering
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 4: Document Categorization / Clustering")
    print("=" * 80)

    categories = lsi.categorize()
    for cat in categories:
        print(f"  📁 {cat['topic_id']} ({cat['assigned_documents_count']} documents) — Top Terms: {cat['summary']}:")
        for d in cat["assigned_documents"]:
            print(f"      • {d['doc_id']}: {d['snippet']}")

    # -------------------------------------------------------------------------
    # Scenario 5: Boundary Condition Validation (Excessive k)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 5: Safe Parameter Clamping (Small Input Validation)")
    print("=" * 80)

    small_collection = [
        "Python backend development with Flask.",
        "Deep learning neural network training with PyTorch.",
    ]
    print(f"Provided {len(small_collection)} documents, but requesting k = 10 topics...")
    res_bounded = run_lsi_analysis(small_collection, num_topics=10)
    print(f"  Requested k : {res_bounded['num_topics_requested']}")
    print(f"  Effective k : {res_bounded['num_topics_used']} (safely clamped)")
    print(f"  Notice      : {res_bounded['k_adjustment_note']}")

    # -------------------------------------------------------------------------
    # Scenario 6: Service API Execution
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 6: Unified NLP Lab Service API Dispatcher (/api/nlp/run)")
    print("=" * 80)

    api_resp = execute_nlp_tool(
        tool_id="lsi_search",
        text="\n\n".join(documents),
        options={
            "mode": "search",
            "query": "transformer neural networks",
            "num_topics": 2,
        },
    )

    print(f"API Tool ID : {api_resp['tool_id']}")
    print(f"Status      : {api_resp['status']}")
    print(f"Success     : {api_resp['success']}")
    print(f"Top Result  : {api_resp['result']['rankings'][0]['doc_id']} ({api_resp['result']['rankings'][0]['similarity_percentage']})")
    print(f"Explanation : {api_resp['result']['explanation']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
