"""Example demonstrating how to use the NLP Lab Text Analysis feature in Crawl.

This runnable example shows:
1. Standard natural language text analysis (language, word count, sentence count, tokens).
2. Multilingual language detection across multiple European and global languages.
3. Handling uncertain language detection when text is brief or ambiguous.
4. Graceful handling of unsupported languages, non-lexical text, and empty inputs.
5. Tokenization customization (lowercasing and stripping punctuation).
6. Invocation via the unified NLP Lab service dispatcher.
"""

from crawl.nlp import (
    analyze_text,
    detect_language,
    tokenize_text,
    split_sentences,
    count_words,
    count_sentences,
    execute_nlp_tool,
)


def main() -> None:
    print("=" * 70)
    print("CRAWL NLP LAB — TEXT ANALYSIS DEMONSTRATION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Scenario 1: Standard Natural Language Text Analysis
    # -------------------------------------------------------------------------
    print("\n--- Scenario 1: Standard English Text Analysis ---")
    english_text = (
        "Natural Language Processing enables computational systems to parse human syntax and semantics. "
        "Dr. Smith developed state-of-the-art pipelines for high-throughput text classification! "
        "How effectively can machines extract structured meaning from unstructured documents?"
    )

    result = analyze_text(english_text)
    print(f"Detected Language : {result['language_name']} ({result['language'].upper()})")
    print(f"Confidence        : {result['language_confidence']:.2f}")
    print(f"Is Uncertain      : {result['is_uncertain']}")
    print(f"Word Count        : {result['word_count']}")
    print(f"Sentence Count    : {result['sentence_count']}")
    print(f"Token Count       : {result['tokens_count']}")
    print(f"First 10 Tokens   : {result['tokens'][:10]}")

    # -------------------------------------------------------------------------
    # Scenario 2: Multilingual Detection
    # -------------------------------------------------------------------------
    print("\n--- Scenario 2: Multilingual Language Detection ---")
    multilingual_samples = {
        "Spanish": "El procesamiento del lenguaje natural permite a las computadoras comprender el lenguaje humano.",
        "French": "Le traitement automatique du langage naturel permet aux ordinateurs de comprendre le langage humain.",
        "German": "Die Verarbeitung natürlicher Sprache ermöglicht es Computern, menschliche Sprache zu verstehen.",
        "Russian": "Обработка естественного языка позволяет компьютерам понимать человеческий язык.",
    }

    for expected_lang, text in multilingual_samples.items():
        res = analyze_text(text)
        print(f"[{expected_lang:8}] -> Code: {res['language'].upper():2} | Name: {res['language_name']:10} | Conf: {res['language_confidence']:.2f} | Words: {res['word_count']:2}")

    # -------------------------------------------------------------------------
    # Scenario 3: Uncertain Language Detection
    # -------------------------------------------------------------------------
    print("\n--- Scenario 3: Uncertain Detection (Short or Ambiguous Text) ---")
    short_text = "Hello"
    short_res = analyze_text(short_text)
    print(f"Input: {short_text!r}")
    print(f"  Status       : {short_res['language_status']}")
    print(f"  Is Uncertain : {short_res['is_uncertain']}")
    print(f"  Note         : {short_res['language_message']}")

    # -------------------------------------------------------------------------
    # Scenario 4: Unsupported Language / Non-Lexical Input
    # -------------------------------------------------------------------------
    print("\n--- Scenario 4: Unsupported / Non-Lexical Input ---")
    gibberish = "xkjdfh skdjfh lksjdfh qowieur zxncbv"
    gib_res = analyze_text(gibberish)
    print(f"Input: {gibberish!r}")
    print(f"  Detected     : {gib_res['language']} ({gib_res['language_name']})")
    print(f"  Status       : {gib_res['language_status']}")
    print(f"  Is Uncertain : {gib_res['is_uncertain']}")
    print(f"  Tokens       : {gib_res['tokens']}")
    print(f"  Word Count   : {gib_res['word_count']}")

    # -------------------------------------------------------------------------
    # Scenario 5: Graceful Empty Input Handling
    # -------------------------------------------------------------------------
    print("\n--- Scenario 5: Graceful Empty Input Handling ---")
    empty_res = analyze_text("")
    print(f"Input: '' (empty)")
    print(f"  Language     : {empty_res['language']}")
    print(f"  Status       : {empty_res['language_status']}")
    print(f"  Word Count   : {empty_res['word_count']}")
    print(f"  Sentence Cnt : {empty_res['sentence_count']}")
    print(f"  Tokens       : {empty_res['tokens']}")

    # -------------------------------------------------------------------------
    # Scenario 6: Token Customization (Lowercase & Strip Punctuation)
    # -------------------------------------------------------------------------
    print("\n--- Scenario 6: Token Customization Options ---")
    phrase = "Crawl's NLP Workbench provides state-of-the-art tools, algorithms, and models!"
    tokens_raw = tokenize_text(phrase)
    tokens_clean = tokenize_text(phrase, lowercase=True, strip_punctuation=True)
    print(f"Original Text   : {phrase}")
    print(f"Default Tokens  : {tokens_raw}")
    print(f"Filtered Tokens : {tokens_clean}")

    # -------------------------------------------------------------------------
    # Scenario 7: Invocation via Unified NLP Lab Service Dispatcher
    # -------------------------------------------------------------------------
    print("\n--- Scenario 7: Unified NLP Lab Service Execution ---")
    payload = execute_nlp_tool(
        tool_id="text_analysis",
        text="The quick brown fox jumps over the lazy dog.",
        options={"lowercase": True, "strip_punctuation": True}
    )
    print(f"Tool ID   : {payload['tool_id']}")
    print(f"Status    : {payload['status']}")
    print(f"Language  : {payload['result']['language_name']} ({payload['result']['language'].upper()})")
    print(f"Words     : {payload['result']['word_count']}")
    print(f"Sentences : {payload['result']['sentence_count']}")
    print(f"Tokens    : {payload['result']['tokens']}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
