"""
title: LiteLLM Manifold Pipe
author: open-webui
date: 2024-05-30
version: 1.0.1
license: MIT
description: A manifold pipe that uses LiteLLM.
requirements: beautifulsoup4, yt_dlp
"""

from bs4 import BeautifulSoup
from functools import lru_cache
from pprint import pformat
from pydantic import BaseModel, Field
from typing import Dict, List, AsyncGenerator, Union, Callable, Any, Awaitable
import aiohttp
import json
import logging
import re
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def load_json(user_value: str, as_list: bool = False) -> Union[Dict, List]:
    user_value = user_value.strip()
    if not user_value:
        return [] if as_list else {}
    try:
        loaded = json.loads(user_value)
        if as_list:
            if not isinstance(loaded, list) or not all(
                isinstance(elem, str) for elem in loaded
            ):
                raise TypeError("Expected a list of strings.")
        elif not isinstance(loaded, dict):
            raise TypeError("Expected a dictionary.")
        return loaded
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Error loading JSON: {e}, Value: {user_value}")
        return [] if as_list else {}

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

class Pipe:
    class Valves(BaseModel):
        LITELLM_BASE_URL: str = Field(
            default="http://litellm.litellm:4000",
            description="(Required) Base URL for LiteLLM.",
        )
        LITELLM_API_KEY: str = Field(
            default="sk-fake-key",
            description="(Required) Your LiteLLM master key.",
        )
        LITELLM_PIPELINE_DEBUG: bool = Field(
            default=False, description="(Optional) Enable debugging."
        )
        EXTRA_METADATA: str = Field(
            default="", description='(Optional) Additional metadata, e.g. {"key": "value"}'
        )
        EXTRA_TAGS: str = Field(
            default='["open-webui"]',
            description='(Optional) A list of tags to apply to requests, e.g. ["open-webui"]',
        )
        YOUTUBE_COOKIES_FILEPATH: str = Field(
            default="",
            description="(Optional) Path to cookies file from youtube.com to aid in title retrieval for citations."
        )
        PERPLEXITY_RETURN_CITATIONS: bool = Field(
            default=True, description="(Optional) Enable citation retrieval for Perplexity models."
        )
        pass

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.current_citations = set()
        self._model_list = None

        if self.valves.LITELLM_PIPELINE_DEBUG:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging is enabled for LiteLLM pipe")

    def _enable_debug(self):
        """
        Set the logging level according to the valve preference.
        """
        if not logger.isEnabledFor(logging.DEBUG):
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging is enabled for LiteLLM pipe")

    def _get_model_list(self):
        """
        Get a (not open-webui compatible) list of models from LiteLLM, including extra
        properties to help us set provider-specific parameters later.
        """
        logger.debug("Fetching model list from LiteLLM")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.LITELLM_API_KEY}",
        }

        try:
            r = requests.get(
                f"{self.valves.LITELLM_BASE_URL}/v1/model/info", headers=headers
            )
            r.raise_for_status()
            models = r.json()

            model_list = []
            for model in models.get("data", []):
                model_info = model.get("model_info", {})
                model_list.append(
                    {
                        "id": model_info.get("id"),
                        "name": model.get("model_name"),
                        "provider": model_info.get("litellm_provider", "openai"),
                    }
                )
                # logger.debug(f"Processed model: {pformat(model_list[-1])}")

            return model_list

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching models from LiteLLM: {e}")
            return []

    async def _build_metadata(self, __user__, __metadata__):
        """
        Construct additional metadata to add to the request.
        This includes trace data to be sent to an observation platform like Langfuse.
        """
        metadata = {
            "tags": set(),
            "trace_user_id": __user__.get("name"),
            "session_id": __metadata__.get("chat_id"),
        }

        extra_metadata = load_json(self.valves.EXTRA_METADATA)
        __metadata__.update(extra_metadata)

        extra_tags = load_json(self.valves.EXTRA_TAGS, as_list=True)
        metadata["tags"].update(extra_tags)
        metadata["tags"] = list(metadata["tags"])

        return metadata

    async def _build_completion_payload(
        self, body: dict, __user__: dict, metadata: dict
    ) -> dict:
        """
        Build the final payload, including the metadata from _build_metadata
        and any provider-specific parameters, using the model list from _get_model_list()
        to identify the provider of the model we're talking to.
        """
        # Required parameters
        model_parts = body["model"].split(".", 1)
        model = model_parts[1] if len(model_parts) > 1 else body["model"]

        logger.debug(f"Body: {body}")
        logger.debug(f"Model name: {model}")

        provider = "openai"  # default
        for m in self._model_list:
            if m["name"] == model or m["id"] == model:
                logger.debug(f"Matched {m['name']} (model id: {m['id']}) with {model}")
                provider = m["provider"]
                break

        # logger.debug(f"Provider for model {model}: {provider}")

        payload = {
            "model": model,
            "messages": body.get("messages", []),
            "stream": True,
            "drop_params": True,
        }

        # Optional parameters with their default values
        optional_openai_params = {
            "seed",
            "temperature",
            "top_p",
            "max_tokens",
            "frequency_penalty",
            "stop",
        }

        # Only add parameters that differ from defaults
        for param in optional_openai_params:
            if param in body:
                payload[param] = body[param]

        # Provider-specific parameters
        provider_params = dict({
            "perplexity": {
                "return_citations": self.valves.PERPLEXITY_RETURN_CITATIONS,
            }
        })

        # Add provider-specific parameters if applicable
        if provider in provider_params:
            logger.debug(f"Adding {provider}-specific parameters")
            for param_name, param_value in provider_params[provider].items():
                if param_value is not None:
                    payload[param_name] = param_value
                    logger.debug(F"Added {param_name}={param_value} from valve settings")
                elif param_name in body:
                    payload[param_name] = body[param_name]
                    logger.debug(f"Added {param_name}={param_value} from body")

        # Add user and metadata if they exist
        if __user__.get("id"):
            payload["user"] = __user__["id"]

        if metadata:
            payload["metadata"] = metadata

        logger.debug(f"Built payload with {len(payload)} parameters")

        return payload

    async def _get_url_title(self, url: str) -> str:
        import yt_dlp as ytdl
        try:
            if "youtube.com" in url or "youtu.be" in url:
                    ydl_opts = {
                        'quiet': True,
                        'no_warnings': True,
                        'extract_flat': True
                    }
                    if hasattr(self.valves, 'YOUTUBE_COOKIES_FILEPATH') and self.valves.YOUTUBE_COOKIES_FILEPATH:
                        ydl_opts['cookiefile'] = self.valves.YOUTUBE_COOKIES_FILEPATH

                    try:
                        with ytdl.YoutubeDL(ydl_opts) as ydl:
                            info = ydl.extract_info(url, download=False)
                            return f"YouTube - {info.get('title', url)}"
                    except Exception as e:
                        logger.warning(f"Failed to retrieve youtube title from url: {url}")
            else:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=2) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, "html.parser")
                            title = soup.title.string if soup.title else None
                            if title:
                                return title.strip()
                        else:
                            logger.warning(f"Failed to retrieve url {url} - status code {response.status}")
        except Exception as e:
            logger.warning(f"Failed to retrieve url: {url}")
            return url

    async def _build_citation_list(self) -> str:
        """
        Take a list of citations and return it as a markup-formatted list:

        [1] [title](url)
        [2] [title](url)
        (...)
        """
        logger.debug(self.current_citations)

        formatted_citations = []
        for i, url in enumerate(self.current_citations, start=1):
            title = await self._get_url_title(url)
            formatted_citations.append(f"[{i}] [{title}]({url})")

        return "\n".join(formatted_citations)

    async def _stream_response(self, response):
        """
        Handle streaming responses.
        """
        self.current_citations.clear()

        async for line in response.content:
            if not line:
                continue

            try:
                line = line.decode("utf-8").strip()
                # logger.debug(f"Raw line: {line}") # this causes a lot of log spam

                if not line or line == "data: ":
                    continue

                if line.startswith("data: "):
                    line = line[6:]

                if line == "[DONE]":
                    continue

                try:
                    json_data = json.loads(line)
                    if "choices" in json_data and len(json_data["choices"]):
                        delta = json_data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield json_data
                    if "citations" in json_data:
                        if isinstance(json_data["citations"], list):
                            self.current_citations.update(json_data["citations"])
                        elif isinstance(json_data["citations"], str):
                            self.current_citations.add(json_data["citations"])
                except json.JSONDecodeError as je:
                    logger.error(f"JSON decode error for line: {line}")
                    logger.error(f"Error details: {str(je)}")
                    continue

            except UnicodeDecodeError as ue:
                logger.error(f"Unicode decode error: {str(ue)}")
                continue

            except Exception as e:
                logger.exception(f"Unexpected error processing line: {str(e)}")
                continue

    async def _get_response(self, response):
        """
        Handle non-streaming responses.
        """
        self.current_citations.clear()

        accumulated_content = ""

        async for line in response.content:
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    try:
                        json_data = json.loads(line[6:])  # Remove 'data:' prefix
                        if "choices" in json_data and json_data["choices"]:
                            delta = json_data["choices"][0].get("delta", {})
                            if "content" in delta:
                                accumulated_content += delta["content"]
                        if "citations" in json_data:
                            if isinstance(json_data["citations"], list):
                                self.current_citations.update(json_data["citations"])
                            elif isinstance(json_data["citations"], str):
                                self.current_citations.add(json_data["citations"])
                    except json.JSONDecodeError:
                        continue

        logger.debug(f"Accumulated content: {accumulated_content}")

        return accumulated_content

    def pipes(self):
        """
        Get the list of models and return it to Open-WebUI.
        """
        logger.debug("pipes() called - fetching model list")
        self._model_list = self._get_model_list()

        model_list = [
            {"id": model["id"], "name": model["name"]} for model in self._model_list
        ]

        return model_list

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __metadata__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
    ) -> AsyncGenerator[str, None]:
        """
        The main pipe through which requests flow.
        """

        if self.valves.LITELLM_PIPELINE_DEBUG:
            self._enable_debug()
            logger.debug("Starting pipe execution")
            logger.debug(f"Metadata: {__metadata__}")
            logger.debug(f"Body: {body}")

        if self._model_list is None:
            logger.warning("Model list not initialized - this shouldn't happen")
            self._model_list = self._get_model_list()

        headers = {"Content-Type": "application/json"}
        if self.valves.LITELLM_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.LITELLM_API_KEY}"

        try:
            emitter = EventEmitter(__event_emitter__)

            metadata = await self._build_metadata(__user__, __metadata__)
            payload = await self._build_completion_payload(body, __user__, metadata)

            logger.debug(f"Final payload: {json.dumps(payload, indent=2)}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=f"{self.valves.LITELLM_BASE_URL}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    try:
                        response.raise_for_status()

                        is_owui_title_gen_task = False
                        if __metadata__.get("task") and len(__metadata__["task"]) > 0:
                            if __metadata__["task"] == "title_generation":
                                is_owui_title_gen_task = True

                        if body["stream"]:
                            async for chunk in self._stream_response(response):
                                yield chunk
                        else:
                            content = await self._get_response(response)
                            yield content

                        # Ensure we have citations to emit; don't add them to the response
                        # to the prompt that generates titles.
                        if self.current_citations and not is_owui_title_gen_task:
                            await emitter.emit(f"Formatting citations...")
                            formatted_citations = await self._build_citation_list()
                            await emitter.emit(
                                status="complete",
                                description="",
                                done=True,
                            )
                            yield f"\n\n{formatted_citations}"

                    except aiohttp.ClientError as e:
                        logger.error(f"Error during request: {e}")
                        yield f"Error: {e}"
        except Exception as e:
            logger.exception(f"Unexpected error in pipe: {e}")
            print(f"Error: {e}")
