"""Base job search adapter interface."""

from typing import List
from crawl.jobs.models import JobQuery, JobResult

class BaseJobAdapter:
    """Interface that must be implemented by each target site crawler/adapter."""

    def __init__(self, name: str) -> None:
        self.name = name

    def search(self, query: JobQuery) -> List[JobResult]:
        """Searches the target site for jobs matching the query constraints.

        Args:
            query: Parsed JobQuery criteria.

        Returns:
            List of normalized JobResult items.
        """
        raise NotImplementedError("Each adapter must implement search()")
