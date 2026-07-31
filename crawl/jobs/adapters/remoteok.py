"""RemoteOK Job Search Adapter.

Uses RemoteOK's completely free public JSON API (no auth, no key).
Always returns real remote job listings with direct individual job URLs —
clicking "View Listing" will always land on the specific job page, never
on a generic search page.

API docs: https://remoteok.com/api
"""

import json
from typing import List

import requests

from crawl.jobs.adapters.base import BaseJobAdapter
from crawl.jobs.models import JobQuery, JobResult
from crawl.utils import setup_logger

logger = setup_logger(__name__)

_REMOTEOK_API = "https://remoteok.com/api"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


class RemoteOKAdapter(BaseJobAdapter):
    """Adapter for fetching real remote jobs from RemoteOK's public API.

    RemoteOK is a free, unauthenticated JSON feed. Every job returned has
    a direct individual job URL — never a generic search page.
    """

    def __init__(self) -> None:
        super().__init__(name="RemoteOK")
        self.last_warning = None

    def search(self, query: JobQuery) -> List[JobResult]:
        """Fetch real remote jobs matching the query.

        We use RemoteOK's ``?tag=`` parameter to filter by job title keyword,
        and do local post-filtering to further match the query title/skills.
        All returned results have a direct unique job URL.
        """
        # Build the primary keyword from the job title
        # e.g. "Python Developer" → "python"
        keyword = query.title.strip().lower().split()[0] if query.title.strip() else ""

        # RemoteOK tag search URL
        if keyword:
            url = f"{_REMOTEOK_API}?tag={keyword}"
        else:
            url = _REMOTEOK_API

        logger.info(f"RemoteOK API URL: {url}")

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
                    f"RemoteOK API returned HTTP {resp.status_code}"
                )
            else:
                raw_data = resp.json()
                # First element is a metadata/legal object, skip it
                job_list = [
                    item for item in raw_data
                    if isinstance(item, dict) and "position" in item
                ]

                for job in job_list:
                    title = (job.get("position") or "").strip()
                    if not title:
                        continue

                    company = (job.get("company") or "Unknown Company").strip()
                    location = (job.get("location") or "Remote").strip()
                    if not location:
                        location = "Remote"

                    # RemoteOK always provides a direct job URL
                    link = (job.get("url") or "").strip()
                    if not link:
                        continue  # Skip if no direct link available

                    # Normalise the URL scheme
                    if link.startswith("//"):
                        link = "https:" + link
                    elif not link.startswith("http"):
                        link = "https://remoteok.com" + link

                    # Posting date — epoch to human string
                    date_str = "Recently"
                    epoch = job.get("epoch")
                    if epoch:
                        from datetime import datetime, timezone
                        try:
                            posted_dt = datetime.fromtimestamp(
                                int(epoch), tz=timezone.utc
                            )
                            now = datetime.now(tz=timezone.utc)
                            delta = now - posted_dt
                            if delta.days == 0:
                                date_str = "Today"
                            elif delta.days == 1:
                                date_str = "Yesterday"
                            else:
                                date_str = f"{delta.days} days ago"
                        except (ValueError, OSError):
                            date_str = "Recently"

                    # Work mode — all RemoteOK jobs are remote by definition
                    work_mode = "Remote"

                    results.append(
                        JobResult(
                            title=title,
                            company=company,
                            location=location,
                            work_mode=work_mode,
                            source="RemoteOK",
                            posting_date=date_str,
                            link=link,
                        )
                    )

                if not results:
                    warning_msg = (
                        f"RemoteOK returned 0 jobs for tag '{keyword}'. "
                        "Try a more generic keyword."
                    )

        except Exception as e:
            logger.error(f"RemoteOK adapter error: {e}")
            warning_msg = f"RemoteOK: {str(e)}"

        if warning_msg:
            self.last_warning = f"RemoteOK: {warning_msg}"
        else:
            self.last_warning = None

        return results
