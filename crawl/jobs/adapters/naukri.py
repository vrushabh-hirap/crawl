"""Naukri.com Job Search Adapter.

Naukri is a React SPA — scraping the HTML page gives only a JS skeleton.
Instead we call Naukri's internal JSON search API (the same endpoint their
own frontend uses), which returns structured job data. Falls back to the
real search URL (not the homepage) when the API is unavailable.
"""

import json
import urllib.parse
from typing import List

from crawl.crawler import Crawl
from crawl.exceptions import CrawlError
from crawl.jobs.adapters.base import BaseJobAdapter
from crawl.jobs.models import JobQuery, JobResult
from crawl.utils import setup_logger

logger = setup_logger(__name__)

# Naukri's internal search API (same one their React SPA calls)
_NAUKRI_API = "https://www.naukri.com/jobapi/v3/search"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.naukri.com/",
    "appid": "109",
    "systemid": "Naukri",
    "x-requested-with": "XMLHttpRequest",
}

# Naukri recency codes
_RECENCY_MAP = {
    "24h": "1",
    "week": "7",
    "month": "30",
}


class NaukriAdapter(BaseJobAdapter):
    """Adapter for fetching jobs from Naukri's internal JSON API."""

    def __init__(self) -> None:
        super().__init__(name="Naukri")

    def search(self, query: JobQuery) -> List[JobResult]:
        # Build the human-facing search URL for fallback links
        title_slug = query.title.lower().replace(" ", "-")
        loc_slug = (query.location or "india").lower().replace(" ", "-")
        search_url = f"https://www.naukri.com/{title_slug}-jobs-in-{loc_slug}"

        # Build API query params
        params: dict = {
            "noOfResults": "30",
            "urlType": "search_by_keyword",
            "searchType": "adv",
            "keyword": query.title,
            "location": query.location or "",
            "k": query.title,
            "l": query.location or "",
            "sort": "r",  # sort by relevance
        }

        if query.recency in _RECENCY_MAP:
            params["experience"] = _RECENCY_MAP[query.recency]

        # Work mode filter
        if query.work_modes:
            wfh_flags = []
            for mode in query.work_modes:
                m = mode.lower()
                if "remote" in m:
                    wfh_flags.append("3")  # 3 = Work from Home on Naukri
                elif "hybrid" in m:
                    wfh_flags.append("9")  # 9 = Hybrid
            if wfh_flags:
                params["wfhType"] = ",".join(wfh_flags)

        api_url = f"{_NAUKRI_API}?{urllib.parse.urlencode(params)}"
        logger.info(f"Naukri API URL: {api_url}")

        results: List[JobResult] = []
        warning_msg = None

        try:
            crawler = Crawl(
                timeout=12.0,
                verify_ssl=False,
                default_headers=_HEADERS,
            )
            page = crawler.fetch(api_url)

            if page.response.status_code in (403, 429, 401):
                warning_msg = (
                    f"Naukri API returned HTTP {page.response.status_code} "
                    "(blocked or requires session auth)"
                )
            else:
                # Parse JSON response
                try:
                    data = json.loads(page.response.text)
                except (json.JSONDecodeError, ValueError):
                    warning_msg = (
                        "Naukri API returned non-JSON response. "
                        "API endpoint may have changed."
                    )
                    data = {}

                job_list = (
                    data.get("jobDetails", [])
                    or data.get("jobs", [])
                    or data.get("results", [])
                    or []
                )

                for job in job_list:
                    title = (
                        job.get("title")
                        or job.get("jobTitle")
                        or ""
                    ).strip()
                    if not title:
                        continue

                    company = (
                        job.get("companyName")
                        or job.get("company", {}).get("label", "")
                        or "Unknown Company"
                    ).strip()

                    location_list = job.get("placeholders", [])
                    location = ""
                    for ph in location_list:
                        if ph.get("type") == "location":
                            location = ph.get("label", "")
                            break
                    if not location:
                        location = (
                            job.get("location")
                            or job.get("locationLabel")
                            or query.location
                            or "India"
                        )

                    link = (
                        job.get("jdURL")
                        or job.get("url")
                        or job.get("applyRedirectURL")
                        or ""
                    )
                    if link and not link.startswith("http"):
                        link = urllib.parse.urljoin("https://www.naukri.com", link)
                    if not link:
                        link = search_url

                    date_str = (
                        job.get("footerPlaceholderLabel")
                        or job.get("modifiedOn")
                        or "Recently"
                    ).strip()

                    # Work mode
                    work_mode = "On-site"
                    wfh = job.get("tagsAndSkills", "") or ""
                    title_lower = title.lower()
                    loc_lower = location.lower()
                    if (
                        "remote" in title_lower
                        or "remote" in loc_lower
                        or "work from home" in wfh.lower()
                        or any("remote" in m.lower() for m in query.work_modes)
                    ):
                        work_mode = "Remote"
                    elif (
                        "hybrid" in title_lower
                        or "hybrid" in loc_lower
                        or any("hybrid" in m.lower() for m in query.work_modes)
                    ):
                        work_mode = "Hybrid"

                    results.append(
                        JobResult(
                            title=title,
                            company=company,
                            location=location,
                            work_mode=work_mode,
                            source="Naukri",
                            posting_date=date_str,
                            link=link,
                        )
                    )

                if not results:
                    warning_msg = (
                        "Naukri API returned 0 jobs for this query. "
                        "Try different keywords or location."
                    )

        except CrawlError as e:
            logger.error(f"Error fetching Naukri page: {e}")
            warning_msg = f"Network or parsing failure: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in Naukri adapter: {e}")
            warning_msg = f"Unexpected error: {str(e)}"

        # When Naukri fails or returns 0 jobs, return an empty list.
        # RemoteOK and Jobicy supply real jobs with direct links — no fake
        # cards with search-page URLs should pollute the results.
        if warning_msg:
            warning_msg = f"Naukri: {warning_msg}"
            self.last_warning = warning_msg
        else:
            self.last_warning = None

        return results

    def _get_mock_data(
        self, query: JobQuery, search_url: str = "https://www.naukri.com/"
    ) -> List[JobResult]:
        """Fallback mock data. Links point to the real Naukri search page."""
        title = query.title or "Software Engineer"
        location = query.location or "Bangalore, KA"
        mode = query.work_modes[0] if query.work_modes else "Remote"

        return [
            JobResult(
                title=f"Associate {title}",
                company="Wipro Technologies",
                location=location,
                work_mode=mode,
                source="Naukri",
                posting_date="1 day ago",
                link=search_url,
            ),
            JobResult(
                title=f"Senior Developer - {title}",
                company="Infosys Digital",
                location=location,
                work_mode=(
                    "Hybrid"
                    if "hybrid" in [m.lower() for m in query.work_modes]
                    else mode
                ),
                source="Naukri",
                posting_date="3 days ago",
                link=search_url,
            ),
            JobResult(
                title=f"Technical Lead ({title})",
                company="Cognizant India",
                location=location,
                work_mode=mode,
                source="Naukri",
                posting_date="5 days ago",
                link=search_url,
            ),
        ]
