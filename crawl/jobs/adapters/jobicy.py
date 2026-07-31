"""Jobicy Job Search Adapter.

Uses Jobicy's completely free public JSON API (no auth, no key required).
Returns real remote tech job listings with direct individual job URLs.

API docs: https://jobicy.com/jobs-rss-feed
"""

import urllib.parse
from typing import List

import requests

from crawl.jobs.adapters.base import BaseJobAdapter
from crawl.jobs.models import JobQuery, JobResult
from crawl.utils import setup_logger

logger = setup_logger(__name__)

_JOBICY_API = "https://jobicy.com/api/v2/remote-jobs"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


class JobicyAdapter(BaseJobAdapter):
    """Adapter for fetching real remote tech jobs from Jobicy's public API.

    Jobicy is a free, unauthenticated JSON API. Every job returned has a
    direct individual job URL — clicking "View Listing" always lands on the
    specific job posting page.
    """

    def __init__(self) -> None:
        super().__init__(name="Jobicy")
        self.last_warning = None

    def search(self, query: JobQuery) -> List[JobResult]:
        """Fetch real remote jobs matching the query from Jobicy.

        Uses the ``?tag=`` and ``?count=`` parameters. All returned results
        have a direct unique job URL that opens the specific posting.
        """
        # Extract the primary keyword from the job title
        keyword = query.title.strip().lower().split()[0] if query.title.strip() else ""

        # Build params
        params: dict = {"count": "20"}
        if keyword:
            params["tag"] = keyword

        url = f"{_JOBICY_API}?{urllib.parse.urlencode(params)}"
        logger.info(f"Jobicy API URL: {url}")

        results: List[JobResult] = []
        warning_msg = None

        try:
            resp = requests.get(
                url,
                headers=_HEADERS,
                verify=False,
                timeout=10,
            )

            if resp.status_code != 200:
                warning_msg = (
                    f"Jobicy API returned HTTP {resp.status_code}"
                )
            else:
                data = resp.json()
                job_list = data.get("jobs", [])

                for job in job_list:
                    title = (job.get("jobTitle") or "").strip()
                    if not title:
                        continue

                    company = (job.get("companyName") or "Unknown Company").strip()

                    # Location: Jobicy uses jobGeo for country / timezone region
                    location = (job.get("jobGeo") or "Remote").strip()
                    if not location:
                        location = "Remote"

                    # Direct job link — always a specific Jobicy listing page
                    link = (job.get("url") or "").strip()
                    if not link:
                        continue  # skip jobs without direct link

                    # Normalise scheme
                    if not link.startswith("http"):
                        link = "https://jobicy.com" + link

                    # Posting date
                    date_str = (job.get("pubDate") or "Recently").strip()
                    # Simplify "2024-07-15 00:00:00" → "Jul 15, 2024"
                    if " " in date_str:
                        try:
                            from datetime import datetime
                            dt = datetime.strptime(date_str.split(" ")[0], "%Y-%m-%d")
                            date_str = dt.strftime("%b %d, %Y")
                        except ValueError:
                            pass

                    # Work mode — jobType can be a list or a string
                    raw_type = job.get("jobType") or []
                    if isinstance(raw_type, list):
                        job_type = " ".join(raw_type).lower()
                    else:
                        job_type = str(raw_type).lower()

                    if "hybrid" in job_type:
                        work_mode = "Hybrid"
                    elif "onsite" in job_type or "on-site" in job_type:
                        work_mode = "On-site"
                    else:
                        work_mode = "Remote"

                    # Skills-based filtering: if user specified skills, only keep
                    # matching jobs (best-effort tag matching)
                    tags = [t.lower() for t in (job.get("jobIndustry") or [])]
                    tags += [t.lower() for t in (job.get("jobType") or [])]
                    if query.skills:
                        # If the job's tags don't mention any of the requested
                        # skills, soft-skip but still include since tag coverage
                        # isn't exhaustive on Jobicy.
                        pass

                    results.append(
                        JobResult(
                            title=title,
                            company=company,
                            location=location,
                            work_mode=work_mode,
                            source="Jobicy",
                            posting_date=date_str,
                            link=link,
                        )
                    )

                if not results:
                    warning_msg = (
                        f"Jobicy returned 0 jobs for tag '{keyword}'. "
                        "Try a more generic keyword."
                    )

        except Exception as e:
            logger.error(f"Jobicy adapter error: {e}")
            warning_msg = f"Jobicy: {str(e)}"

        if warning_msg:
            self.last_warning = f"Jobicy: {warning_msg}"
        else:
            self.last_warning = None

        return results
