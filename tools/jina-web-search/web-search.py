"""
title: Web Search using SearXNG and jina.ai
author: samling with heavy inspiration from constLiakos, justinh-rahb and ther3zz
funding_url: https://github.com/open-webui
version: 0.1.0
license: MIT
"""

import asyncio
import concurrent.futures
import re
import requests
import json
import unicodedata
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from typing import Callable, Any
from urllib.parse import urlparse


class HelperFunctions:
    def __init__(self):
        pass

    def get_base_url(self, url):
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        return base_url

    def generate_excerpt(self, content, max_length=200):
        return content[:max_length] + "..." if len(content) > max_length else content

    def format_text(self, original_text):
        soup = BeautifulSoup(original_text, "html.parser")
        formatted_text = soup.get_text(separator=" ", strip=True)
        formatted_text = unicodedata.normalize("NFKC", formatted_text)
        formatted_text = re.sub(r"\s+", " ", formatted_text)
        formatted_text = formatted_text.strip()
        formatted_text = self.remove_emojis(formatted_text)
        return formatted_text

    def remove_emojis(self, text):
        return "".join(c for c in text if not unicodedata.category(c).startswith("So"))

    def process_search_result(self, result, valves):
        title_site = self.remove_emojis(result["title"])
        url_site = result["url"]
        snippet = result.get("content", "")

        # Check if the website is in the ignored list, but only if IGNORED_WEBSITES is not empty
        if valves.SEARXNG_IGNORED_WEBSITES:
            base_url = self.get_base_url(url_site)
            if any(
                ignored_site.strip() in base_url
                for ignored_site in valves.SEARXNG_IGNORED_WEBSITES.split(",")
            ):
                return None

        try:
            response_site = requests.get(url_site, timeout=20)
            response_site.raise_for_status()
            html_content = response_site.text

            soup = BeautifulSoup(html_content, "html.parser")
            content_site = self.format_text(soup.get_text(separator=" ", strip=True))

            truncated_content = self.truncate_to_n_words(
                content_site, valves.PAGE_CONTENT_WORDS_LIMIT
            )

            return {
                "title": title_site,
                "url": url_site,
                "content": truncated_content,
                "snippet": self.remove_emojis(snippet),
            }

        except requests.exceptions.RequestException as e:
            return None

    def process_scrape(self, result, valves, user_valves):
        functions = HelperFunctions()

        # await emitter.emit(f"Scraping {result['url']}")
        jina_url = f"https://r.jina.ai/{result['url']}"

        headers = {
            "X-No-Cache": "true" if valves.JINA_DISABLE_CACHING else "false",
            "X-With-Generated-Alt": "true",
            "X-Retain-Images": "none",
        }

        if user_valves.JINA_API_KEY:
            headers["Authorization"] = f"Bearer {user_valves.JINA_API_KEY}"
        elif valves.JINA_GLOBAL_API_KEY:
            headers["Authorization"] = f"Bearer {valves.JINA_GLOBAL_API_KEY}"

        try:
            response = requests.get(jina_url, headers=headers, timeout=120)
            response.raise_for_status()

            should_clean = user_valves.JINA_CLEAN_CONTENT

            content = functions.clean_urls(response.text) if should_clean else response.text

            title = functions.extract_title(content)

            content = response.text
            if valves.PAGE_CONTENT_WORDS_LIMIT != 0:
                content = functions.truncate_to_n_words(
                    content, valves.PAGE_CONTENT_WORDS_LIMIT
                )

            return {
                "title": title,
                "url": jina_url,
                "content": content,
                "excerpt": functions.generate_excerpt(content),
            }

        except requests.exceptions.RequestException as e:
            return {
                    "title": jina_url,
                    "url": jina_url,
                    "content": f"Failed to retrieve the page. Error: {str(e)}",
                    "excerpt": ""
                }

    def extract_title(self, text):
        """
        Extracts the title from a string containing structured text.

        :param text: The input string containing the title.
        :return: The extracted title string, or None if the title is not found.
        """
        match = re.search(r"Title: (.*)\n", text)
        return match.group(1).strip() if match else None

    def clean_urls(self, text) -> str:
        """
        Cleans URLs from a string containing structured text.

        :param text: The input string containing the URLs.
        :return: The cleaned string with URLs removed.
        """
        return re.sub(r"\((http[^)]+)\)", "", text)

    def truncate_to_n_words(self, text, token_limit):
        tokens = text.split()
        truncated_tokens = tokens[:token_limit]
        return " ".join(truncated_tokens)


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def emit(self, description="Unknown State", status="in_progress", done=False):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )


class Tools:
    class Valves(BaseModel):
        SEARXNG_ENGINE_API_BASE_URL: str = Field(
            default="https://example.com/search",
            description="The base URL for Search Engine",
        )
        SEARXNG_IGNORED_WEBSITES: str = Field(
            default="",
            description="Comma-separated list of websites to ignore",
        )
        SEARXNG_RETURNED_SCRAPED_PAGES_NO: int = Field(
            default=5,
            description="The number of search engine results to parse",
        )
        SEARXNG_SCRAPED_PAGES_NO: int = Field(
            default=3,
            description="Total pages scraped.",
        )
        JINA_DISABLE_CACHING: bool = Field(
            default=False, description="Bypass Jina Cache when scraping"
        )
        JINA_GLOBAL_API_KEY: str = Field(
            default="",
            description="(Optional) Jina API key. Allows a higher rate limit when scraping. Used when a User-specific API key is not available.",
        )
        PAGE_CONTENT_WORDS_LIMIT: int = Field(
            default=5000,
            description="Limit words content for each page.",
        )
        CITATION_LINKS: bool = Field(
            default=True,
            description="If True, send custom citations with links",
        )

    class UserValves(BaseModel):
        JINA_CLEAN_CONTENT: bool = Field(
            default=True,
            description="Remove links and image urls from scraped content. This reduces the number of tokens.",
        )
        JINA_API_KEY: str = Field(
            default="",
            description="(Optional) Jina API key. Allows a higher rate limit when scraping.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }

    async def search_web(
        self,
        query: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Search the web and get the content of the relevant pages. Search for unknown knowledge, news, info, public contact info, weather, etc.
        :params query: Web Query used in search engine.
        :return: The content of the pages in json format.
        """
        functions = HelperFunctions()
        emitter = EventEmitter(__event_emitter__)

        await emitter.emit(f"Initiating web search for: {query}")

        search_engine_url = self.valves.SEARXNG_ENGINE_API_BASE_URL

        # Ensure RETURNED_SCRAPED_PAGES_NO does not exceed SCRAPED_PAGES_NO
        if (
            self.valves.SEARXNG_RETURNED_SCRAPED_PAGES_NO
            > self.valves.SEARXNG_SCRAPED_PAGES_NO
        ):
            self.valves.SEARXNG_RETURNED_SCRAPED_PAGES_NO = (
                self.valves.SEARXNG_SCRAPED_PAGES_NO
            )

        params = {
            "q": query,
            "format": "json",
            "number_of_results": self.valves.SEARXNG_RETURNED_SCRAPED_PAGES_NO,
        }

        try:
            await emitter.emit("Sending request to search engine")
            resp = requests.get(
                search_engine_url, params=params, headers=self.headers, timeout=120
            )

            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            limited_results = results[: self.valves.SEARXNG_SCRAPED_PAGES_NO]
            await emitter.emit(f"Retrieved {len(limited_results)} search results")

        except requests.exceptions.RequestException as e:
            await emitter.emit(
                status="error",
                description=f"Error during search: {str(e)}",
                done=True,
            )
            return json.dumps({"error": str(e)})

        # Process the results
        search_results_json = []
        if limited_results:
            await emitter.emit(f"Processing search results")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        functions.process_search_result, result, self.valves
                    )
                    for result in limited_results
                ]
                for future in concurrent.futures.as_completed(futures):
                    result_json = future.result()
                    if result_json:
                        try:
                            json.dumps(result_json)
                            search_results_json.append(result_json)
                        except (TypeError, ValueError):
                            continue
                    if (
                        len(search_results_json)
                        >= self.valves.SEARXNG_RETURNED_SCRAPED_PAGES_NO
                    ):
                        break

            if self.valves.CITATION_LINKS and __event_emitter__:
                for result in search_results_json:
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "document": [result["content"]],
                                "metadata": [{"source": result["url"]}],
                                "source": {"name": result["title"]},
                            },
                        },
                    )

        await emitter.emit(
            status="complete",
            description=f"Web search completed. Retrieved content from {len(search_results_json)} pages",
            done=True,
        )

        scrape_results_json = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    functions.process_scrape, result, self.valves, self.user_valves
                )
                for result in search_results_json
            ]
            for future in concurrent.futures.as_completed(futures):
                result_json = future.result()
                if result_json:
                    try:
                        json.dumps(result_json)
                        scrape_results_json.append(result_json)
                    except (TypeError, ValueError):
                        continue
                if (
                    len(scrape_results_json)
                    >= self.valves.SEARXNG_RETURNED_SCRAPED_PAGES_NO
                ):
                    break

        return json.dumps(scrape_results_json, ensure_ascii=False)


async def main():
    tools = Tools()
    results = await tools.search_web("Who won the most recent celtics game")

    print(results)


if __name__ == "__main__":
    asyncio.run(main())