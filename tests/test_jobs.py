"""Unit tests for the Job Search backend engine and adapters."""

import unittest
from crawl.jobs.models import JobQuery, JobResult
from crawl.jobs.engine import JobSearchEngine
from crawl.jobs.adapters.linkedin import LinkedInAdapter
from crawl.jobs.adapters.indeed import IndeedAdapter
from crawl.jobs.adapters.naukri import NaukriAdapter

class TestJobSearch(unittest.TestCase):
    """Verifies that dataclasses, adapters, and search engine function correctly."""

    def test_query_and_result_models(self) -> None:
        """Tests that JobQuery and JobResult models initialize and serialize correctly."""
        query = JobQuery(
            title="Backend Engineer",
            work_modes=["Remote", "Hybrid"],
            location="Remote",
            skills=["Python", "Flask"],
            recency="week"
        )
        self.assertEqual(query.title, "Backend Engineer")
        self.assertEqual(query.location, "Remote")
        self.assertEqual(query.recency, "week")
        self.assertEqual(query.work_modes, ["Remote", "Hybrid"])
        
        q_dict = query.to_dict()
        self.assertEqual(q_dict["title"], "Backend Engineer")
        self.assertEqual(q_dict["skills"], ["Python", "Flask"])

        result = JobResult(
            title="Python Developer",
            company="Tech Corp",
            location="Chicago, IL",
            work_mode="Remote",
            source="LinkedIn",
            posting_date="2 days ago",
            link="https://linkedin.com/test-job"
        )
        r_dict = result.to_dict()
        self.assertEqual(r_dict["title"], "Python Developer")
        self.assertEqual(r_dict["company"], "Tech Corp")
        self.assertEqual(r_dict["source"], "LinkedIn")

    def test_adapters_mock_generation(self) -> None:
        """Tests that mock data generation logic works correctly for all adapters."""
        query = JobQuery(title="Data Analyst", work_modes=["Remote"])
        
        li = LinkedInAdapter()
        li_mock = li._get_mock_data(query)
        self.assertTrue(len(li_mock) > 0)
        self.assertEqual(li_mock[0].source, "LinkedIn")
        
        ind = IndeedAdapter()
        ind_mock = ind._get_mock_data(query)
        self.assertTrue(len(ind_mock) > 0)
        self.assertEqual(ind_mock[0].source, "Indeed")
        
        nk = NaukriAdapter()
        nk_mock = nk._get_mock_data(query)
        self.assertTrue(len(nk_mock) > 0)
        self.assertEqual(nk_mock[0].source, "Naukri")

    def test_engine_search(self) -> None:
        """Tests that the JobSearchEngine runs searches across registered adapters concurrently."""
        engine = JobSearchEngine()
        query = JobQuery(title="Frontend Engineer", work_modes=["Remote"])
        results, warnings = engine.search(query)
        
        self.assertTrue(len(results) > 0)
        for r in results:
            self.assertIsInstance(r, JobResult)
            self.assertIn(r.source, ["LinkedIn", "Indeed", "Naukri"])
            self.assertTrue(len(r.title) > 0)
            self.assertTrue(len(r.company) > 0)
            self.assertTrue(len(r.link) > 0)

if __name__ == "__main__":
    unittest.main()
