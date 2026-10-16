"""Example demonstrating N-gram Language Modeling in Crawl's NLP Lab.

This runnable example demonstrates:
1. Building an N-gram language model from a text corpus.
2. Supporting configurable Bigrams (n=2) and Trigrams (n=3).
3. Calculating joint probability and perplexity for a word sequence.
4. Explaining and comparing unseen sequence handling:
   - Unsmoothed Maximum Likelihood Estimation (MLE): unseen transitions cause total probability
     to collapse to 0.0 and perplexity to Infinity.
   - Laplace (+1) Smoothing: redistributes probability mass, assigning (C + 1) / (C_context + V)
     so unseen transitions maintain non-zero probabilities.
   - Add-k smoothing (e.g. k=0.5).
5. Inspecting step-by-step Markov transitions and formula counts for an assignment demonstration.
6. Dispatching via Crawl's unified NLP Lab service.
"""

from crawl.nlp import (
    NGramLanguageModel,
    calculate_sequence_probability,
    execute_nlp_tool,
)


def main() -> None:
    print("=" * 80)
    print("CRAWL NLP LAB — N-GRAM STATISTICAL LANGUAGE MODEL DEMONSTRATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Training Corpus
    # -------------------------------------------------------------------------
    corpus = (
        "to be or not to be that is the question whether tis nobler in the mind "
        "to suffer the slings and arrows of outrageous fortune or to take arms "
        "against a sea of troubles and by opposing end them"
    )

    print("\n--- Training Corpus ---")
    print(f'"{corpus}"')

    # -------------------------------------------------------------------------
    # Scenario 1: Bigram Language Model (n=2) with Laplace Smoothing
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 1: Bigram Language Model (n=2) — Sequence Likelihood")
    print("=" * 80)

    target_phrase = "to be or not to be"
    print(f'Target Sequence to Evaluate: "{target_phrase}"\n')

    bigram_res = calculate_sequence_probability(
        corpus=corpus,
        target_sequence=target_phrase,
        n=2,
        smoothing="laplace",
    )

    print(f"Model Order      : {bigram_res['order_name']} (n={bigram_res['n']})")
    print(f"Smoothing        : {bigram_res['smoothing'].title()} (+1 / +V)")
    print(f"Vocabulary Size  : |V| = {bigram_res['vocabulary_size']} unique words")
    print(f"Corpus Tokens    : {bigram_res['corpus_tokens_count']} tokens")
    print(f"Joint P(W)       : {bigram_res['total_probability_scientific']} ({bigram_res['total_probability']:.8e})")
    print(f"log2 P(W)        : {bigram_res['log2_probability']}")
    print(f"Perplexity (PP)  : {bigram_res['perplexity']}")
    print(f"Unseen N-grams   : {bigram_res['unseen_ngram_count']} transitions")

    print("\nStep-by-Step Markov Transition Probabilities:")
    print(f"{'Step':4} | {'Target Word':12} | {'Context':10} | {'Counts (C/Cctx)':16} | {'P(w|ctx)':10} | {'Status':8}")
    print("-" * 75)
    for s in bigram_res["steps"]:
        count_str = f"{s['ngram_count']}/{s['context_count']}"
        print(f"{s['step']:<4} | {s['target_word']:12} | {s['context_str']:10} | {count_str:16} | {s['probability_formatted']:10} | {s['status'].upper():8}")

    print("\nFormula Demonstration for Step 2:")
    step2 = bigram_res["steps"][1]
    print(f"  {step2['formula_str']}")

    # -------------------------------------------------------------------------
    # Scenario 2: Trigram Language Model (n=3)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 2: Trigram Language Model (n=3) — Higher-Order Context")
    print("=" * 80)

    trigram_res = calculate_sequence_probability(
        corpus=corpus,
        target_sequence=target_phrase,
        n=3,
        smoothing="laplace",
    )

    print(f"Model Order      : {trigram_res['order_name']} (n={trigram_res['n']})")
    print(f"Joint P(W)       : {trigram_res['total_probability_scientific']}")
    print(f"Perplexity (PP)  : {trigram_res['perplexity']}")

    print("\nTrigram Transition Steps:")
    for s in trigram_res["steps"]:
        print(f"  Step {s['step']}: {s['formula_str']} -> p={s['probability_formatted']} [{s['status']}]")

    # -------------------------------------------------------------------------
    # Scenario 3: Handling Unseen Sequences — The Problem of Zero Probability
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 3: Handling Unseen Sequences — Smoothing vs. MLE Collapse")
    print("=" * 80)

    unseen_phrase = "quantum physics computer"
    print(f'Unseen Target Sequence: "{unseen_phrase}"\n')

    # A. Unsmoothed MLE (smoothing="none")
    mle_res = calculate_sequence_probability(
        corpus=corpus,
        target_sequence=unseen_phrase,
        n=2,
        smoothing="none",
    )
    print("► Method A: Maximum Likelihood Estimation (No Smoothing / MLE)")
    print(f"  Joint P(W)      : {mle_res['total_probability_scientific']}")
    print(f"  Perplexity      : {mle_res['perplexity']}")
    print(f"  Unseen N-grams  : {mle_res['unseen_ngram_count']} / {len(mle_res['steps'])}")
    print(f"  Diagnosis       : If any C(ngram) == 0, the entire joint product collapses to 0.0.")
    print(f"  Perplexity      : 1 / P(W)^(1/N) = 1 / 0 = Infinity (model is infinitely perplexed).")

    # B. Laplace (+1) Smoothing
    laplace_unseen = calculate_sequence_probability(
        corpus=corpus,
        target_sequence=unseen_phrase,
        n=2,
        smoothing="laplace",
    )
    print("\n► Method B: Laplace (+1) Smoothing")
    print(f"  Joint P(W)      : {laplace_unseen['total_probability_scientific']} ({laplace_unseen['total_probability']:.6e})")
    print(f"  Perplexity      : {laplace_unseen['perplexity']}")
    print(f"  Unseen N-grams  : {laplace_unseen['unseen_ngram_count']} / {len(laplace_unseen['steps'])}")
    print(f"  Solution        : Formula P = (0 + 1) / (C_ctx + V) > 0 prevents numerical collapse.")
    for s in laplace_unseen["steps"]:
        print(f"    • {s['formula_str']} -> p={s['probability_formatted']} [{s['status']}]")

    # C. Add-k Smoothing (k=0.5)
    add_k_unseen = calculate_sequence_probability(
        corpus=corpus,
        target_sequence=unseen_phrase,
        n=2,
        smoothing="add_k",
        k=0.5,
    )
    print("\n► Method C: Add-k (Lidstone) Smoothing (k=0.5)")
    print(f"  Joint P(W)      : {add_k_unseen['total_probability_scientific']}")
    print(f"  Perplexity      : {add_k_unseen['perplexity']}")

    # -------------------------------------------------------------------------
    # Scenario 4: Object-Oriented NGramLanguageModel API
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 4: Object-Oriented NGramLanguageModel Interface")
    print("=" * 80)

    model = NGramLanguageModel(n=2, smoothing="laplace")
    model.fit(corpus)
    print(f"Trained model with {len(model.vocab)} vocab items and {sum(model.context_counts.values())} n-grams.")
    res_oo = model.calculate_sequence_probability("whether tis nobler")
    print(f"Evaluated 'whether tis nobler': P(W) = {res_oo['total_probability_scientific']}, PP = {res_oo['perplexity']}")

    # -------------------------------------------------------------------------
    # Scenario 5: Unified Service Dispatcher Execution
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 5: NLP Lab Service API Dispatcher (/api/nlp/run)")
    print("=" * 80)

    api_resp = execute_nlp_tool(
        tool_id="ngram_probability",
        text=corpus,
        options={
            "n": 2,
            "smoothing": "laplace",
            "target_sequence": "to be or not to be",
        },
    )
    print(f"API Tool ID : {api_resp['tool_id']}")
    print(f"Status      : {api_resp['status']}")
    print(f"Success     : {api_resp['success']}")
    print(f"Order       : {api_resp['result']['order_name']}")
    print(f"Probability : {api_resp['result']['total_probability_scientific']}")
    print(f"Explanation : {api_resp['result']['explanation']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
