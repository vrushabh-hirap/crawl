"""Unit tests for the 7-stage ATS Pipeline."""

import pytest
from crawl.jobs.ats import ATSAnalyzer
from crawl.jobs.ats.pipeline.stage1_ingestion import ingest
from crawl.jobs.ats.pipeline.stage2_segmentation import segment_resume, segment_jd
from crawl.jobs.ats.pipeline.stage3_normalizer import normalize, NormalizerConfig
from crawl.jobs.ats.pipeline.stage4_ner import extract_entities
from crawl.jobs.ats.pipeline.stage5_skills import match_skills
from crawl.jobs.ats.pipeline.stage6_semantic import compute_semantic_similarity
from crawl.jobs.ats.pipeline.stage7_reporter import build_report


SAMPLE_RESUME = """
John Doe
john.doe@email.com | +1 (555) 123-4567 | linkedin.com/in/johndoe | github.com/johndoe

SUMMARY
Experienced Senior Software Engineer with expertise in building scalable microservices,
cloud-native architectures, and machine learning pipelines.

EXPERIENCE
Senior Software Engineer | Acme Corp
Jan 2021 – Present
- Reduced API response latency by 40% using Redis caching and PostgreSQL query optimization.
- Led a cross-functional team of 6 engineers to deliver Kubernetes microservices on AWS.
- Developed CI/CD pipelines using GitHub Actions, cutting deployment time by 50%.

Software Developer | Tech Corp
June 2018 – December 2020
- Built RESTful APIs using Python, Flask, and Docker.
- Improved database performance across 15+ microservices.

EDUCATION
B.S. Computer Science | State University
Graduated 2018 | GPA: 3.8/4.0

SKILLS
Python, JavaScript, TypeScript, K8s, Docker, AWS, PostgreSQL, Redis, CI/CD, REST API, Git
"""

SAMPLE_JD = """
About the Role:
We are looking for a Senior Python Developer with strong backend experience.

Requirements:
- Minimum 3+ years of experience with Python, Docker, and Kubernetes (K8s).
- Hands-on experience with AWS, PostgreSQL, and building REST APIs.
- Experience with CI/CD pipelines and microservices architecture.

Preferred Qualifications:
- Experience with React or TypeScript is a plus.
- Knowledge of Machine Learning (ML) pipelines.
"""


def test_stage1_ingestion_plain_text():
    result = ingest(SAMPLE_RESUME.encode("utf-8"), "resume.txt")
    assert "John Doe" in result.text
    assert result.metadata["format"] == "txt"
    assert isinstance(result.parse_hazards, list)


def test_stage2_segmentation():
    resume_map = segment_resume(SAMPLE_RESUME)
    assert "Summary" in resume_map
    assert "Experience" in resume_map
    assert "Education" in resume_map
    assert "Skills" in resume_map

    jd_map = segment_jd(SAMPLE_JD)
    assert "Requirements" in jd_map or "Preferred" in jd_map or "Body" in jd_map


def test_stage3_normalizer():
    norm_res = normalize(SAMPLE_RESUME, NormalizerConfig(method="lemmatizer"))
    assert "python" in norm_res.tokens
    assert "docker" in norm_res.tokens or "kubernetes" in norm_res.expanded_text

    # Test stemmer mode fallback
    norm_stem = normalize(SAMPLE_RESUME, NormalizerConfig(method="stemmer"))
    assert len(norm_stem.tokens) > 0



def test_stage4_ner():
    resume_map = segment_resume(SAMPLE_RESUME)
    entities = extract_entities(resume_map, SAMPLE_RESUME)

    assert len(entities.work_history) >= 1
    assert len(entities.education) >= 1
    assert entities.total_exp_months > 0
    assert entities.years_of_experience > 0.0


def test_stage5_skills():
    resume_map = segment_resume(SAMPLE_RESUME)
    jd_map = segment_jd(SAMPLE_JD)

    res_norm = normalize(SAMPLE_RESUME)
    jd_norm = normalize(SAMPLE_JD)

    skills_result = match_skills(res_norm, jd_norm, jd_map)

    assert len(skills_result.hard_matched) > 0
    assert skills_result.overall_keyword_score > 0
    # Check synonym match resolution: K8s in resume matching Kubernetes in JD
    has_k8s_synonym = any(
        m.is_synonym or m.jd_term == "kubernetes" or m.resume_term in ("k8s", "kubernetes")
        for m in skills_result.hard_matched
    )
    assert has_k8s_synonym


def test_stage6_semantic_graceful():
    sim = compute_semantic_similarity(SAMPLE_RESUME, SAMPLE_JD)
    if sim is not None:
        assert 0.0 <= sim.overall_similarity <= 1.0
        assert sim.available is True


def test_full_pipeline_analyzer():
    analyzer = ATSAnalyzer()
    report = analyzer.analyze(SAMPLE_RESUME.encode("utf-8"), "resume.txt", SAMPLE_JD)

    assert "ats_score" in report
    assert "grade" in report
    assert report["ats_score"] > 0
    assert "breakdown" in report
    assert "keywords" in report
    assert "matched" in report["keywords"]
    assert "missing" in report["keywords"]
    assert "entities" in report
    assert "recommendations" in report
    assert len(report["recommendations"]) > 0

    # Ensure recommendations are structured dicts
    for rec in report["recommendations"]:
        assert isinstance(rec, dict)
        assert "category" in rec
        assert "severity" in rec
        assert "text" in rec
