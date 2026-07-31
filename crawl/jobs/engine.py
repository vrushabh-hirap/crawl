"""Concurrent engine to run multiple job site crawls in parallel."""

import concurrent.futures
from typing import List, Tuple
from crawl.jobs.models import JobQuery, JobResult
from crawl.jobs.adapters import (
    LinkedInAdapter,
    IndeedAdapter,
    NaukriAdapter,
    RemoteOKAdapter,
    JobicyAdapter,
)
from crawl.utils import setup_logger
logger = setup_logger(__name__)

class JobSearchEngine:
    """Coordinates search across multiple job site adapters concurrently."""

    def __init__(self) -> None:
        # Register all adapters.
        # RemoteOK and Jobicy use free, open APIs that always return real job
        # listings with direct individual job URLs — no mock fallback needed.
        self.adapters = [
            LinkedInAdapter(),
            IndeedAdapter(),
            NaukriAdapter(),
            RemoteOKAdapter(),
            JobicyAdapter(),
        ]

    def search(self, query: JobQuery) -> Tuple[List[JobResult], List[str]]:
        """Executes search across all registered job adapters concurrently.

        Args:
            query: Parsed JobQuery constraints.

        Returns:
            A tuple of (accumulated JobResults list, warnings messages list).
        """
        all_results: List[JobResult] = []
        warnings: List[str] = []

        # Run concurrent crawls using a ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.adapters)) as executor:
            # Map each adapter search to a future
            future_to_adapter = {
                executor.submit(adapter.search, query): adapter 
                for adapter in self.adapters
            }

            for future in concurrent.futures.as_completed(future_to_adapter):
                adapter = future_to_adapter[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    # Check if the adapter encountered rate limits or parse failures
                    if hasattr(adapter, "last_warning") and adapter.last_warning:
                        warnings.append(adapter.last_warning)
                except Exception as e:
                    logger.error(f"Adapter {adapter.name} raised an unhandled exception: {e}")
                    warnings.append(f"{adapter.name} search failed unexpectedly: {str(e)}")

        # Sort combined results: e.g., recently posted jobs first if parsing is reliable
        return all_results, warnings
