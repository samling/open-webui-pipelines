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
from copy import deepcopy
from functools import lru_cache
from pydantic import BaseModel, Field
from typing import Dict, List, AsyncGenerator, Union, Callable, Any, Awaitable
import aiohttp
import json
import logging
import os
import requests

# as function
#from open_webui.utils.misc import get_last_user_message, pop_system_message

# as pipeline
from utils.pipelines.main import get_last_user_message, pop_system_message

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

    async def emit_message(self, content=""):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "message",
                    "data": {
                        "content": content
                    }
                }
            )

    async def emit_status(self, description="Unknown State", status="in_progress", done=False):
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

    async def emit_source(self, name: str, document: str, url: str, html: bool = True):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "source",
                    "data": {
                        "document": [document],
                        "metadata": [
                            {
                                "source": name,
                                "html": html
                            }
                        ],
                        "source": {
                            "name": name,
                            "url": url
                        }
                    }
                }
            )

    async def emit_citation(self, source: str, metadata: str, document: str):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "citation",
                    "data": {
                        "document": document,
                        "metadata": metadata,
                        "source": source
                    }
                }
            )

class Pipeline:
    class Valves(BaseModel):
        LITELLM_BASE_URL: str = ""
        LITELLM_API_KEY: str = ""
        LITELLM_PIPELINE_DEBUG: bool = False
        EXTRA_METADATA: str = ""
        EXTRA_TAGS: str = '["open-webui"]',
        REQUEST_TIMEOUT: int = 5
        YOUTUBE_COOKIES_FILEPATH: str = ""
        VISION_ROUTER_ENABLED: bool =  False
        VISION_MODEL_ID: str = ""
        SKIP_REROUTE_MODELS: str = ""
        PERPLEXITY_RETURN_CITATIONS: bool = True
        PERPLEXITY_RETURN_IMAGES: bool = False
        PERPLEXITY_RETURN_RELATED_QUESTIONS: bool = False

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves(
            **{
                "LITELLM_BASE_URL": os.getenv("LITELLM_BASE_URL", "http://litellm.litellm:4000"),
                "LITELLM_API_KEY": os.getenv("LITELLM_API_KEY", "sk-fake-key"),
                "LITELLM_PIPELINE_DEBUG": os.getenv("LITELLM_PIPELINE_DEBUG", False),
                "EXTRA_METADATA": os.getenv("EXTRA_METADATA", ''),
                "EXTRA_TAGS": os.getenv("EXTRA_TAGS", '["open-webui"]'),
                "REQUEST_TIMEOUT": os.getenv("REQUEST_TIMEOUT", 5),
                "YOUTUBE_COOKIES_FILEPATH": os.getenv("YOUTUBE_COOKIES_FILEPATH", ""),
                "VISION_ROUTER_ENABLED": os.getenv("VISION_ROUTER_ENABLED", False),
                "VISION_MODEL_ID": os.getenv("VISION_MODEL_ID", ""),
                "SKIP_REROUTE_MODELS": os.getenv("SKIP_REROUTE_MODELS", ""),
                "PERPLEXITY_RETURN_CITATIONS": os.getenv("PERPLEXITY_RETURN_CITATIONS", True),
                "PERPLEXITY_RETURN_IMAGES": os.getenv("PERPLEXITY_RETURN_IMAGES", False),
                "PERPLEXITY_RETURN_RELATED_QUESTIONS": os.getenv("PERPLEXITY_RETURN_RELATED_QUESTIONS", False),
            }
        )
        self.pipelines = self._get_pipeline_model_list()
        pass

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        global logger
        if self.valves.LITELLM_PIPELINE_DEBUG:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging is enabled for LiteLLM pipe")
        else:
            logger.setLevel(logging.INFO)
            logger.info("Debug logging is disabled for LiteLLM pipe")

        self.pipelines = self._get_pipeline_model_list()
        pass

    async def on_valves_updated(self):
        global logger
        if self.valves.LITELLM_PIPELINE_DEBUG:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging is enabled for LiteLLM pipe")
        else:
            logger.setLevel(logging.INFO)
            logger.info("Debug logging is disabled for LiteLLM pipe")

        self.pipelines = self._get_pipeline_model_list()
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

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

    def _get_pipeline_model_list(self):
        logger.debug("pipes() called - fetching model list")
        self._model_list = self._get_model_list()

        model_list = [
            # use model['name'] instead of model['id'] for 'id' here because that's easier to match on
            {"id": model["name"], "name": model["name"]} for model in self._model_list
        ]

        return model_list

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
        self,
        body: dict,
        __user__: dict,
        metadata: dict,
        emitter: EventEmitter
    ) -> dict:
        """
        Build the final payload, including the metadata from _build_metadata
        and any provider-specific parameters, using the model list from _get_model_list()
        to identify the provider of the model we're talking to.
        """
        # Required parameters
        model_parts = body["model"].split(".", 1) # models from this pipe look like e.g. "litellm.gpt-4o"
        model_prefix = model_parts[0] if len(model_parts) > 1 else "" # "litellm"
        model_name = model_parts[1] if len(model_parts) > 1 else body["model"] # "gpt-4o"

        # Get the most recent user message
        messages = body.get("messages", [])
        last_user_message_content = get_last_user_message(messages)
        if last_user_message_content is None:
            return body

        # Process system prompt if there is one
        system_message, messages = pop_system_message(messages)
        system_prompt = "You are a helpful assistant."
        if system_message is not None:
            system_prompt = system_message["content"]

        # Check for images in the last user message by inspecting the messages directly
        has_images = False
        for message in reversed(messages):
            if message["role"] == "user":
                if isinstance(message.get("content"), list):
                    has_images = any(
                        item.get("type") == "image_url" for item in message["content"]
                    )
                break

        # Set the model to the vision model if it's defined and there's an image in the most recent user message
        if has_images:
            logger.debug(f"Found image in last user message; attempting to reroute request to vision model")
            logger.debug(f"SKIPPED MODELS: {self.valves.SKIP_REROUTE_MODELS}")
            logger.debug(f"CURRENT MODEL NAME: {model_name}")

            if self.valves.VISION_MODEL_ID:
                # If we're not already using the rerouted models and we're not using a model that we want to skip rerouting in, reroute it
                if model_name != self.valves.VISION_MODEL_ID and model_name not in self.valves.SKIP_REROUTE_MODELS:
                    model_name = self.valves.VISION_MODEL_ID
                    await emitter.emit_status(description=f"Request routed to {self.valves.VISION_MODEL_ID}", done=True)

        # Clean base64-encoded images from previous messages
        cleaned_messages = []
        for message in messages:
            cleaned_message = message.copy()
            if isinstance(message.get("content"), list):
                if (message["role"] == "user" and 
                    any(item.get("type") == "text" and item.get("text") == last_user_message_content
                        for item in message["content"])):
                    # Keep the current message intact
                    cleaned_message = message
                else:
                    # Clean up old messages
                    cleaned_content = []
                    for content in message["content"]:
                        if content.get("type") == "image_url" and "url" in content["image_url"]:
                            if content["image_url"]["url"].startswith("data:image"):
                                content = {
                                    "type": "text",
                                    "text": "[Previous Image]"
                                }
                        cleaned_content.append(content)
                    cleaned_message["content"] = cleaned_content
            cleaned_messages.append(cleaned_message)

        # Determine which provider the request is going to in order to add any provider-specific metadata
        provider = "openai"  # default
        for m in self._model_list:
            if m["name"] == model_name or m["id"] == model_name:
                logger.debug(f"Matched {m['name']} (model id: {m['id']}) with {model_name}")
                provider = m["provider"]
                break

        # Optional parameters with their default values
        optional_openai_params = {
            "seed",
            "temperature",
            "top_p",
            "max_tokens",
            "frequency_penalty",
            "stop",
        }

        # Provider-specific parameters
        provider_params = dict({
            "perplexity": {
                "return_citations": self.valves.PERPLEXITY_RETURN_CITATIONS,
                "return_images": self.valves.PERPLEXITY_RETURN_IMAGES,
                "return_related_questions": self.valves.PERPLEXITY_RETURN_RELATED_QUESTIONS,
            }
        })

        # Final payload with base properties
        payload = {
            "model": model_name, # optional vision model if image in last message
            "messages": [{"role": "system", "content": system_prompt}, *cleaned_messages], # only most recent message contains data blobs
            "stream": True,
            "drop_params": True, # litellm
        }

        # Only add openai parameters that differ from defaults
        for param in optional_openai_params:
            if param in body:
                payload[param] = body[param]

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

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

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
                    try:
                        async with session.get(url, headers=headers, timeout=self.valves.REQUEST_TIMEOUT) as response:
                            if response.status == 200:
                                html = await response.text()
                                soup = BeautifulSoup(html, "html.parser")

                                # Try metadata title first
                                meta_title = soup.find("meta", property="og:title")
                                if meta_title and meta_title.get("content"):
                                    return meta_title["content"].strip()
                                
                                # Fall back to regular title
                                if soup.title and soup.title.string:
                                    return soup.title.string.strip()

                                # Finally, try h1
                                if soup.h1:
                                    return soup.h1.get_text().strip()
                            else:
                                logger.warning(f"Failed to retrieve url: {url} (status code {response.status})")
                    except aiohttp.ClientError as e:
                        logger.warning(f"Initial request failed, trying with different User-Agent: {str(e)}")

                        # Try with a mobile user agent
                        headers['User-Agent'] = 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1'

                        try:
                            async with session.get(url, headers=headers, timeout=self.valves.REQUEST_TIMEOUT) as response:
                                if response.status == 200:
                                    html = await response.text()
                                    soup = BeautifulSoup(html, "html.parser")

                                    # Try metadata title first
                                    meta_title = soup.find("meta", property="og:title")
                                    if meta_title and meta_title.get("content"):
                                        return meta_title["content"].strip()
                                    
                                    # Fall back to regular title
                                    if soup.title and soup.title.string:
                                        return soup.title.string.strip()

                                    # Finally, try h1
                                    if soup.h1:
                                        return soup.h1.get_text().strip()
                        except Exception as e2:
                            logger.warning(f"Both attempts failed for url {url}: {str(e2)}")

        except Exception as e:
            logger.warning(f"Failed to retrieve url: {url}")

        return url

    async def _build_citation_list(self, citations: set) -> str:
        """
        Take a list of citations and return it as a list.
        """
        citations_list = []
        for i, url in enumerate(citations, start=1):
            title = await self._get_url_title(url)
            citation = {
                "title": title,
                "url": url,
                "content": ""
            }
            citations_list.append(citation)
            logger.debug(f"Appended citation: {citation}")

        return citations_list

    async def _process_citations(self, citations: set, emitter: EventEmitter, is_title_gen: bool = False):
        """
        Process and emit citations if any exist and we're not generating a title.
        """
        if citations and not is_title_gen:
            await emitter.emit_status(description=f"Formatting citations...")

            citations_list = await self._build_citation_list(citations)

            await emitter.emit_message(content=f"\n<details>\n<summary>Sources</summary>")
            for i, citation in enumerate(citations_list, 1):
                await emitter.emit_message(
                    content=f"\n[{i}] [{citation.get('title')}]({citation.get('url')})"
                )
            await emitter.emit_message(content=f"\n</details>\n")
            await emitter.emit_status(description="", done=True, status="complete")

    async def _stream_response(self, response):
        """
        Handle streaming responses.
        """

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
                    except json.JSONDecodeError:
                        continue

        logger.debug(f"Accumulated content: {accumulated_content}")

        return accumulated_content

    async def pipe(
        self,
        user_message: str,
        model_ide: str,
        messages: List[dict],
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
    ) -> AsyncGenerator[str, None]:
        """
        The main pipe through which requests flow.
        """

        citations = set()

        if self._model_list is None:
            logger.warning("Model list not initialized - this shouldn't happen")
            self._model_list = self._get_model_list()

        headers = {"Content-Type": "application/json"}
        if self.valves.LITELLM_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.LITELLM_API_KEY}"

        try:
            emitter = EventEmitter(__event_emitter__)

            metadata = await self._build_metadata(__user__, __metadata__)
            payload = await self._build_completion_payload(body, __user__, metadata, emitter)

            # Remove binary data from the logs to keep things clean
            log_payload = deepcopy(payload)
            if "messages" in log_payload:
                for message in log_payload["messages"]:
                    if isinstance(message.get("content"), list):
                        for content in message["content"]:
                            if content.get("type") == "image_url" and "url" in content["image_url"]:
                                if content["image_url"]["url"].startswith("data:image"):
                                    content["image_url"]["url"] = "[BASE64_IMAGE_DATA]"
            logger.debug(f"Final payload: {json.dumps(log_payload, indent=2)}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=f"{self.valves.LITELLM_BASE_URL}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    try:
                        response.raise_for_status()

                        is_title_gen = __metadata__.get("task") == "title_generation"

                        if body["stream"]:
                            async for chunk in self._stream_response(response):
                                if "citations" in chunk:
                                    if isinstance(chunk["citations"], list):
                                        citations.update(chunk["citations"])
                                    elif isinstance(chunk["citations"], str):
                                        citations.add(chunk["citations"])
                                yield chunk
                        else:
                            content = await self._get_response(response)
                            yield content

                        await self._process_citations(citations, emitter, is_title_gen)

                        await emitter.emit_status(
                            status="complete",
                            description="",
                            done=True,
                        )

                    except aiohttp.ClientError as e:
                        logger.error(f"Error during request: {e}")
                        yield f"Error: {e}"
        except Exception as e:
            logger.exception(f"Unexpected error in pipe: {e}")