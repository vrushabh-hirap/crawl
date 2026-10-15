"""ATS Pipeline subpackage.

Seven independently-testable stages that together form the production ATS engine.

Stage 1 — Document Ingestion  (stage1_ingestion.py)
Stage 2 — Section Segmentation (stage2_segmentation.py)
Stage 3 — Text Normalization  (stage3_normalizer.py)
Stage 4 — Named Entity Recognition (stage4_ner.py)
Stage 5 — Skill & Keyword Matching (stage5_skills.py)
Stage 6 — Semantic Similarity  (stage6_semantic.py)
Stage 7 — Scoring & Report Generation (stage7_reporter.py)
"""
