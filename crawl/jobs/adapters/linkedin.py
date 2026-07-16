"""LinkedIn Job Search Adapter.

Uses LinkedIn's public Jobs Guest API which returns JSON job data without
requiring a login or JavaScript rendering. Falls back to the search URL
(not the homepage) when the API is unavailable.
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

# LinkedIn's public (unauthenticated) Jobs Guest API endpoint
_LINKEDIN_API = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"

# Realistic browser headers to avoid trivial bot-detection
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.linkedin.com/jobs/search/",
}


class LinkedInAdapter(BaseJobAdapter):
    """Adapter for fetching jobs from LinkedIn's public Jobs Guest API."""

    def __init__(self) -> None:
        super().__init__(name="LinkedIn")

    def search(self, query: JobQuery) -> List[JobResult]:
        # Build query parameters for the API
        params: dict = {
            "keywords": query.title,
            "location": query.location or "Worldwide",
            "start": "0",
        }

        # Recency filter
        if query.recency == "24h":
            params["f_TPR"] = "r86400"
        elif query.recency == "week":
            params["f_TPR"] = "r604800"
        elif query.recency == "month":
            params["f_TPR"] = "r2592000"

        # Work mode filter (f_WT: 1=Onsite, 2=Remote, 3=Hybrid)
        if query.work_modes:
            modes = []
            for mode in query.work_modes:
                m = mode.lower().strip()
                if "remote" in m:
                    modes.append("2")
                elif "onsite" in m or "on-site" in m:
                    modes.append("1")
                elif "hybrid" in m:
                    modes.append("3")
            if modes:
                params["f_WT"] = modes[0]

        api_url = f"{_LINKEDIN_API}?{urllib.parse.urlencode(params)}"

        # Also build the human-readable search URL for fallback links
        search_url = (
            f"https://www.linkedin.com/jobs/search/?{urllib.parse.urlencode(params)}"
        )
        logger.info(f"LinkedIn API URL: {api_url}")

        results: List[JobResult] = []
        warning_msg = None

        try:
            crawler = Crawl(
                timeout=10.0,
                verify_ssl=False,
                default_headers=_HEADERS,
            )
            page = crawler.fetch(api_url)

            if page.response.status_code in (400, 403, 429, 999):
                warning_msg = (
                    f"LinkedIn API returned HTTP {page.response.status_code} "
                    "(rate-limited or blocked)"
                )
            else:
                # The API returns an HTML fragment of <li> job cards
                job_cards = page.find_all("li")

                for card in job_cards:
                    # Title
                    title_node = card.find("h3.base-search-card__title")
                    if not title_node:
                        title_node = card.find("h3")
                    if not title_node:
                        continue
                    title = title_node.get_text().strip()
                    if not title:
                        continue

                    # Company
                    company_node = card.find("h4.base-search-card__subtitle")
                    if not company_node:
                        company_node = card.find("h4")
                    company = (
                        company_node.get_text().strip()
                        if company_node
                        else "Unknown Company"
                    )

                    # Location
                    loc_node = card.find("span.job-search-card__location")
                    location = (
                        loc_node.get_text().strip()
                        if loc_node
                        else (query.location or "Remote")
                    )

                    # Direct job link — LinkedIn API cards contain a full absolute href
                    link_node = card.find("a.base-card__full-link")
                    if not link_node:
                        link_node = card.find("a")
                    link = (
                        link_node.get_attribute("href") if link_node else ""
                    ) or ""
                    if link and not link.startswith("http"):
                        link = urllib.parse.urljoin(
                            "https://www.linkedin.com", link
                        )
                    # Strip tracking params to get a clean job URL
                    if link and "?" in link:
                        clean = link.split("?")[0]
                        link = clean if "/jobs/view/" in clean else link
                    if not link:
                        link = search_url

                    # Date
                    date_node = card.find("time")
                    date_str = (
                        date_node.get_text().strip() if date_node else "Recently"
                    )

                    # Work mode inference
                    work_mode = "On-site"
                    title_lower = title.lower()
                    loc_lower = location.lower()
                    if "remote" in title_lower or "remote" in loc_lower:
                        work_mode = "Remote"
                    elif "hybrid" in title_lower or "hybrid" in loc_lower:
                        work_mode = "Hybrid"
                    elif query.work_modes:
                        for m in query.work_modes:
                            if "remote" in m.lower():
                                work_mode = "Remote"
                            elif "hybrid" in m.lower():
                                work_mode = "Hybrid"

                    results.append(
                        JobResult(
                            title=title,
                            company=company,
                            location=location,
                            work_mode=work_mode,
                            source="LinkedIn",
                            posting_date=date_str,
                            link=link,
                        )
                    )

                if not results:
                    warning_msg = (
                        "No job cards found in LinkedIn API response. "
                        "Page structure may have changed."
                    )

        except CrawlError as e:
            logger.error(f"Error fetching LinkedIn: {e}")
            warning_msg = f"Network or parsing failure: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in LinkedIn adapter: {e}")
            warning_msg = f"Unexpected error: {str(e)}"

        # Fallback: use search URL as the link (not the homepage)
        if warning_msg and not results:
            logger.info("Serving mock fallback data for LinkedIn.")
            results = self._get_mock_data(query, search_url)
            warning_msg = f"LinkedIn: {warning_msg} (Showing mock results for demo)"
            self.last_warning = warning_msg
        else:
            self.last_warning = None

        return results

    def _get_mock_data(
        self, query: JobQuery, search_url: str = "https://www.linkedin.com/jobs/search/"
    ) -> List[JobResult]:
        """Fallback mock data. Links point to the real search results page."""
        title = query.title or "Software Engineer"
        location = query.location or "New York, NY"
        mode = query.work_modes[0] if query.work_modes else "Remote"

        return [
            JobResult(
                title=f"Senior {title}",
                company="TechCorp Solutions",
                location=location,
                work_mode=mode,
                source="LinkedIn",
                posting_date="2 hours ago",
                link=search_url,
            ),
            JobResult(
                title=f"Lead {title} (React/Node)",
                company="Global Logistics Inc.",
                location=location,
                work_mode=(
                    "Hybrid"
                    if "hybrid" in [m.lower() for m in query.work_modes]
                    else mode
                ),
                source="LinkedIn",
                posting_date="1 day ago",
                link=search_url,
            ),
            JobResult(
                title=f"Junior {title}",
                company="Quantum Labs",
                location=location,
                work_mode=mode,
                source="LinkedIn",
                posting_date="3 days ago",
                link=search_url,
            ),
        ]
