"""Indeed Job Search Adapter.

Indeed aggressively blocks bots (HTTP 403). We send realistic browser headers
and disable SSL verification. When blocked, fallback links point to the actual
search URL so the user can visit Indeed directly for the same query.
"""

import urllib.parse
from typing import List

from crawl.crawler import Crawl
from crawl.exceptions import CrawlError
from crawl.jobs.adapters.base import BaseJobAdapter
from crawl.jobs.models import JobQuery, JobResult
from crawl.utils import setup_logger

logger = setup_logger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
}


class IndeedAdapter(BaseJobAdapter):
    """Adapter for crawling jobs from Indeed."""

    def __init__(self) -> None:
        super().__init__(name="Indeed")

    def search(self, query: JobQuery) -> List[JobResult]:
        # Build search URL
        params: dict = {
            "q": query.title,
            "l": query.location or "",
        }

        if query.recency == "24h":
            params["fromage"] = "1"
        elif query.recency == "week":
            params["fromage"] = "7"
        elif query.recency == "month":
            params["fromage"] = "30"

        query_str = urllib.parse.urlencode(params)

        # Remote work filter
        sc_filter = ""
        if query.work_modes:
            for mode in query.work_modes:
                if "remote" in mode.lower():
                    sc_filter = "&sc=0kf%3Aattr%28DSQF7%29%3B"
                    break

        url = f"https://www.indeed.com/jobs?{query_str}{sc_filter}"
        logger.info(f"Indeed Search URL: {url}")

        results: List[JobResult] = []
        warning_msg = None

        try:
            crawler = Crawl(
                timeout=10.0,
                verify_ssl=False,
                default_headers=_HEADERS,
            )
            page = crawler.fetch(url)

            if page.response.status_code in (403, 429):
                logger.warning(
                    f"Indeed returned status {page.response.status_code} "
                    "(blocked/rate-limited)."
                )
                warning_msg = (
                    f"Request was rate-limited/blocked "
                    f"(HTTP {page.response.status_code})"
                )
            else:
                # Indeed renders job cards server-side in some regions;
                # try multiple known selectors
                job_cards = page.find_all("div.job_seen_beacon")
                if not job_cards:
                    job_cards = page.find_all("td.resultContent")
                if not job_cards:
                    job_cards = page.find_all("div.cardOutline")
                if not job_cards:
                    job_cards = page.find_all("div.jobsearch-SerpJobCard")

                for card in job_cards:
                    title_node = card.find("h2.jobTitle")
                    if not title_node:
                        title_node = card.find("h2")
                    if not title_node:
                        continue
                    title = title_node.get_text().strip()
                    if not title:
                        continue

                    company_node = card.find("span.companyName")
                    if not company_node:
                        company_node = card.find("span.company")
                    company = (
                        company_node.get_text().strip()
                        if company_node
                        else "Unknown Company"
                    )

                    loc_node = card.find("div.companyLocation")
                    if not loc_node:
                        loc_node = card.find("div.location")
                    location = (
                        loc_node.get_text().strip()
                        if loc_node
                        else (query.location or "United States")
                    )

                    # Build direct job link
                    link_node = card.find("a")
                    link = (
                        link_node.get_attribute("href") if link_node else ""
                    ) or ""
                    if link and not link.startswith("http"):
                        link = urllib.parse.urljoin("https://www.indeed.com", link)
                    if not link:
                        link = url

                    date_node = card.find("span.date")
                    date_str = (
                        date_node.get_text().strip().replace("Posted", "").strip()
                        if date_node
                        else "Recently"
                    )

                    work_mode = "On-site"
                    if (
                        "remote" in location.lower()
                        or "remote" in title.lower()
                        or any("remote" in m.lower() for m in query.work_modes)
                    ):
                        work_mode = "Remote"
                    elif (
                        "hybrid" in location.lower()
                        or "hybrid" in title.lower()
                        or any("hybrid" in m.lower() for m in query.work_modes)
                    ):
                        work_mode = "Hybrid"

                    results.append(
                        JobResult(
                            title=title,
                            company=company,
                            location=location,
                            work_mode=work_mode,
                            source="Indeed",
                            posting_date=date_str,
                            link=link,
                        )
                    )

                if not results:
                    warning_msg = (
                        "No job cards found. Indeed may require JavaScript "
                        "rendering or your IP may be rate-limited."
                    )

        except CrawlError as e:
            logger.error(f"Error fetching Indeed page: {e}")
            warning_msg = f"Network or parsing failure: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in Indeed adapter: {e}")
            warning_msg = f"Unexpected error: {str(e)}"

        # When Indeed is blocked or returns no cards, return an empty list.
        # RemoteOK and Jobicy supply real jobs with direct links — no fake
        # cards with search-page URLs should pollute the results.
        if warning_msg:
            warning_msg = f"Indeed: {warning_msg}"
            self.last_warning = warning_msg
        else:
            self.last_warning = None

        return results

    def _get_mock_data(
        self, query: JobQuery, search_url: str = "https://www.indeed.com/jobs"
    ) -> List[JobResult]:
        """Fallback mock data. Links point to the real search results page."""
        title = query.title or "Software Engineer"
        location = query.location or "Austin, TX"
        mode = query.work_modes[0] if query.work_modes else "Remote"

        return [
            JobResult(
                title=f"Full Stack {title}",
                company="Innovate Solutions Ltd",
                location=location,
                work_mode=mode,
                source="Indeed",
                posting_date="5 hours ago",
                link=search_url,
            ),
            JobResult(
                title=f"{title} II",
                company="Nexus Finance Tech",
                location=location,
                work_mode=mode,
                source="Indeed",
                posting_date="2 days ago",
                link=search_url,
            ),
            JobResult(
                title=f"Staff {title} (Cloud Architecture)",
                company="Apex Systems",
                location=location,
                work_mode=(
                    "Hybrid"
                    if "hybrid" in [m.lower() for m in query.work_modes]
                    else mode
                ),
                source="Indeed",
                posting_date="4 days ago",
                link=search_url,
            ),
        ]
