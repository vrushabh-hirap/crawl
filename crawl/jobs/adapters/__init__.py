"""Adapters for individual job search websites."""

from crawl.jobs.adapters.base import BaseJobAdapter
from crawl.jobs.adapters.linkedin import LinkedInAdapter
from crawl.jobs.adapters.indeed import IndeedAdapter
from crawl.jobs.adapters.naukri import NaukriAdapter

__all__ = [
    "BaseJobAdapter",
    "LinkedInAdapter",
    "IndeedAdapter",
    "NaukriAdapter"
]
