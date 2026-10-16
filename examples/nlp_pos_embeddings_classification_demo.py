"""Example demonstrating POS tagging, dense text embeddings, and classification in Crawl.

This runnable example shows:
1. Grammatical Part-of-Speech (POS) tagging with tokens, fine-grained tags, and descriptions.
2. Dense text embedding generation with vector dimensionality and optional sentence-transformers support.
3. Multi-class text categorization evaluated against a documented benchmark dataset.
4. User-provided dataset flow for custom classification tasks.
5. Invoking operations via the unified Crawl NLP Lab service layer.
"""

from crawl.nlp import (
    pos_tag_text,
    get_text_embedding,
    classify_text,
    execute_nlp_tool,
)


def main() -> None:
    print("=" * 76)
    print("CRAWL NLP LAB — POS, EMBEDDINGS & CLASSIFICATION DEMONSTRATION")
    print("=" * 76)

    sample_sentence = (
        "The senior software engineer deployed high-throughput microservices using "
        "Docker and Kubernetes on cloud infrastructure clusters."
    )

    # -------------------------------------------------------------------------
    # Scenario 1: Part-of-Speech Tagging
    # -------------------------------------------------------------------------
    print("\n--- Scenario 1: Part-of-Speech Tagging ---")
    pos_results = pos_tag_text(sample_sentence)

    print(f"Engine          : {pos_results['engine']}")
    print(f"Total Tokens    : {pos_results['tokens_count']}")
    print(f"Lexical Density : {pos_results['lexical_density'] * 100:.1f}% (Content / total words)")
    print(f"POS Distribution: {pos_results['pos_distribution']}")
    print("\nTagged Token Sample:")
    print(f"  {'Token':15} | {'POS':6} | {'Tag':5} | {'Description'}")
    print("  " + "-" * 55)
    for t in pos_results["tagged_tokens"][:10]:
        print(f"  {t['token']:15} | {t['pos']:6} | {t['tag']:5} | {t['description']}")

    # -------------------------------------------------------------------------
    # Scenario 2: Dense Text Embeddings
    # -------------------------------------------------------------------------
    print("\n--- Scenario 2: Dense Text Embeddings ---")
    emb_results = get_text_embedding(sample_sentence)

    print(f"Model Engine    : {emb_results['model']}")
    print(f"Dimensions      : {emb_results['dimensions']}")
    print(f"Vector L2 Norm  : {emb_results['vector_norm']}")
    print(f"Summary Stats   : {emb_results['stats']}")
    print(f"Vector Preview  : {emb_results['vector_preview']}")
    if emb_results.get("notice"):
        print(f"\nModel Notice    :\n  {emb_results['notice']}")

    # -------------------------------------------------------------------------
    # Scenario 3: Text Classification (Documented Benchmark Dataset)
    # -------------------------------------------------------------------------
    print("\n--- Scenario 3: Text Classification (Benchmark Corpus) ---")
    cls_results = classify_text(sample_sentence)

    print(f"Input Text      : {sample_sentence!r}")
    print(f"Predicted Class : {cls_results['predicted_label']}")
    print(f"Confidence      : {cls_results['confidence_percentage']}")
    print(f"Explanation     : {cls_results['explanation']}")

    print("\nCategory Confidence Ranking:")
    for rank_item in cls_results["ranking"]:
        matches = f" (Matches: {', '.join(rank_item['matched_keywords'])})" if rank_item['matched_keywords'] else ""
        print(f"  #{rank_item['rank']} {rank_item['label']:12} : {rank_item['percentage']:>6}{matches}")

    print("\nEvaluation Details:")
    eval_info = cls_results["evaluation"]
    print(f"  Model Architecture    : {eval_info['model_architecture']}")
    print(f"  Dataset Name          : {eval_info['dataset_type']}")
    print(f"  Training Categories   : {eval_info['training_categories']}")
    print(f"  Exemplar Documents    : {eval_info['training_documents_total']}")
    print(f"  Benchmark Accuracy    : {eval_info['benchmark_accuracy'] * 100:.0f}%")
    print(f"  Margin Over Runner-up : +{eval_info['margin_over_runner_up'] * 100:.1f}%")

    # -------------------------------------------------------------------------
    # Scenario 4: User-Provided Dataset Flow
    # -------------------------------------------------------------------------
    print("\n--- Scenario 4: User-Provided Custom Dataset Flow ---")
    custom_dataset = {
        "Aerospace": [
            "Rocket propulsion systems, orbital trajectory calculations, and satellite launch vehicles.",
            "Aerodynamic drag coefficients, supersonic wind tunnel simulations, and wing lift analysis.",
        ],
        "Biochemistry": [
            "Enzyme kinetics, protein crystallography, and peptide bond synthesis in cellular matrices.",
            "Lipid bilayer permeability, mitochondrial ATP generation, and metabolic pathway assays.",
        ],
    }

    custom_input = "The experimental satellite payload reached stable low-earth orbital trajectory."
    custom_res = classify_text(custom_input, dataset=custom_dataset)

    print(f"Input Text      : {custom_input!r}")
    print(f"Predicted Class : {custom_res['predicted_label']}")
    print(f"Confidence      : {custom_res['confidence_percentage']}")
    print(f"Dataset Type    : {custom_res['evaluation']['dataset_type']}")
    print(f"Categories      : {custom_res['evaluation']['training_categories']}")

    # -------------------------------------------------------------------------
    # Scenario 5: Unified Service Dispatcher Execution
    # -------------------------------------------------------------------------
    print("\n--- Scenario 5: Unified NLP Lab Service Execution ---")
    payload = execute_nlp_tool(
        tool_id="pos_embeddings_classification",
        text="Senior machine learning engineer training neural networks on GPU clusters.",
        options={"task": "classification"},
    )
    print(f"Success         : {payload['success']}")
    print(f"Tool ID         : {payload['tool_id']}")
    print(f"Status          : {payload['status']}")
    print(f"Prediction      : {payload['result']['predicted_label']} ({payload['result']['confidence_percentage']})")

    print("\n" + "=" * 76)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 76)


if __name__ == "__main__":
    main()
