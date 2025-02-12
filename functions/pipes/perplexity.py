"""
title: Perplexity Manifold Pipe
author: samling, based on pipe by justinh-rahb and moblangeois
author_url: https://github.com/samling/open-webui-pipelines
version: 0.1.0
license: MIT
requirements: beautifulsoup4, yt_dlp
"""

from bs4 import BeautifulSoup
from functools import lru_cache
from pprint import pformat
from pydantic import BaseModel, Field
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Union,
    )
from utils.misc import get_last_user_message
from utils.misc import pop_system_message

import aiohttp
import json
import logging
import re

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


class Pipe:
    class Valves(BaseModel):
        NAME_PREFIX: str = Field(
            default="Perplexity.",
            description="The prefix applied before the model names.",
        )
        PERPLEXITY_API_BASE_URL: str = Field(
            default="https://api.perplexity.ai",
            description="The base URL for Perplexity API endpoints.",
        )
        PERPLEXITY_API_KEY: str = Field(
            default="",
            description="Required API key to access Perplexity services.",
        )
        PIPE_DEBUG: bool = Field(
            default=False, description="(Optional) Enable debugging for the pipe."
        )
        EXTRA_METADATA: str = Field(
            default="{}", description='(Optional) Additional metadata, e.g. {"key": "value"}'
        )
        EXTRA_TAGS: str = Field(
            default='["open-webui"]',
            description='(Optional) A list of tags to apply to requests, e.g. ["open-webui"]',
        )
        REQUEST_TIMEOUT: int = Field(
            default=5,
            description="(Optional) Timeout (in seconds) for aiohttp session requests. Default is 5s."
        )
        YOUTUBE_COOKIES_FILEPATH: str = Field(
            default="path/to/cookies.txt",
            description="(Optional) Path to cookies file from youtube.com to aid in title retrieval for citations."
        )
        PERPLEXITY_RETURN_CITATIONS: bool = Field(
            default=True, description="(Optional) Enable citation retrieval for Perplexity models."
        )
        PERPLEXITY_RETURN_IMAGES: bool = Field(
            default=False, description="(Optional) Enable image retrieval for Perplexity models. Note: This is a beta feature."
        )
        PERPLEXITY_RETURN_RELATED_QUESTIONS: bool = Field(
            default=False, description="(Optional) Enable related question retrieval for Perplexity models. Note: This is a beta feature."
        )

    class UserValves(BaseModel):
        EXTRA_METADATA: str = Field(
            default="{}", description='(Optional) Additional metadata, e.g. {"key": "value"}'
        )
        EXTRA_TAGS: str = Field(
            default='["open-webui"]',
            description='(Optional) A list of tags to apply to requests, e.g. ["open-webui"]',
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()

    def _parse_model_string(self, model_id):
        """
        Parse model ID into manifold and model name components

        Inputs:
            model_id: str
            In the instance of a manifold pipe, this is usually something like
            {pipe_name}.{model_name}, e.g. "perplexity_manifold.llama-3.1..."

        Outputs:

        """
        parts = model_id.split(".", 1)
        manifold_name = None
        model_name = None

        if len(parts) == 1:
            model_name = parts[0]
        else:
            manifold_name = parts[0]
            model_name = parts[1]

        logger.debug(f"Manifold name: {manifold_name}")
        logger.debug(f"Model name: {model_name}")

        return manifold_name, model_name

    async def _build_metadata(self, __user__, __metadata__, user_valves):
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

        logger.debug(f"User valves: {user_valves}")
        logger.debug(f"User metadata: {user_valves.EXTRA_METADATA}")
        extra_user_metadata = load_json(user_valves.EXTRA_METADATA)
        __metadata__.update(extra_user_metadata)

        logger.debug(f"User tags: {user_valves.EXTRA_TAGS}")
        extra_tags = load_json(self.valves.EXTRA_TAGS, as_list=True)
        metadata["tags"].update(extra_tags)

        extra_user_tags = load_json(user_valves.EXTRA_TAGS, as_list=True)
        metadata["tags"].update(extra_user_tags)

        metadata["tags"] = list(metadata["tags"])

        return metadata

    async def _build_completion_payload(
        self,
        body: dict,
        __user__: dict,
        metadata: dict,
        user_valves: UserValves,
        emitter: EventEmitter
    ) -> dict:
        """
        Build the final payload, including the metadata from _build_metadata
        """
        # Models from this pipe look like e.g. "{manifold_name}.gpt-4o" or "{manifold_name}.anthropic/claude-3.5-sonnet"
        logger.debug(f"Model from open-webui request: {body['model']}")
        manfold_name, model_name = self._parse_model_string(body['model'])

        # Get the most recent user message
        messages = body.get("messages", [])
        last_user_message_content = get_last_user_message(messages)
        if last_user_message_content is None:
            return body

        # Process system prompt if there is one
        system_message, messages = pop_system_message(messages)
        system_prompt = "You are a helpful assistant."
        if system_message is not None:
            logger.debug(f"Using non-default system prompt: {system_message['content']}")
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
            if self.valves.VISION_MODEL_ID and self.valves.VISION_MODEL_ID is not "fake-reroute-model":
                # If we're not already using the rerouted models and we're not using a model that we want to skip rerouting in, reroute it
                if model_name != self.valves.VISION_MODEL_ID and model_name not in self.valves.SKIP_REROUTE_MODELS:
                    model_name = self.valves.VISION_MODEL_ID
                    await emitter.emit_status(description=f"Request routed to {self.valves.VISION_MODEL_ID}", done=True)
                else:
                    logger.debug(f"Model is the same as target vision model or is in SKIP_REROUTE_MODELS")

        # Clean base64-encoded images from previous messages
        logger.debug(f"Stripping encoded image data from past messages")
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

        # Trim messages to fit in model's max_tokens
        # logger.debug(f"Trimming message content to max_input_tokens value: {litellm_model_props['model_info']['max_input_tokens']}")
        # cleaned_messages = trim_messages(cleaned_messages, model_name)

        # Optional parameters with their default values
        optional_openai_params = {
            "seed",
            "temperature",
            "top_p",
            "max_tokens",
            "frequency_penalty",
            "stop",
        }

        # Final payload with base properties
        payload = {
            "model": model_name, # optional vision model if image in last message
            "messages": [{"role": "system", "content": system_prompt}, *cleaned_messages], # only most recent message contains data blobs
            "stream": body.get("stream", True),
            "return_citations": self.valves.PERPLEXITY_RETURN_CITATIONS,
            "return_images": self.valves.PERPLEXITY_RETURN_IMAGES,
            "return_related_questions": self.valves.PERPLEXITY_RETURN_RELATED_QUESTIONS,

        }

        # Only add openai parameters that differ from defaults
        for param in optional_openai_params:
            if param in body:
                payload[param] = body[param]

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
                    if hasattr(self.valves, 'YOUTUBE_COOKIES_FILEPATH') and self.valves.YOUTUBE_COOKIES_FILEPATH and self.valves.YOUTUBE_COOKIES_FILEPATH is not "path/to/cookies.txt":
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

            await emitter.emit_message(content=f"\n\n<details>\n<summary>Sources</summary>")
            for i, citation in enumerate(citations_list, 1):
                await emitter.emit_message(
                    content=f"\n[{i}] [{citation.get('title')}]({citation.get('url')})"
                )
            await emitter.emit_message(content=f"\n</details>\n")
            await emitter.emit_status(description="", done=True, status="complete")

    def _convert_citations_to_superscript(self, match):
        """
        Convert a citation like [1] to its Unicode superscript equivalent
        """
        # Mapping of normal numbers to their superscript versions
        superscript_map = {
            '0': '⁰',
            '1': '¹',
            '2': '²',
            '3': '³',
            '4': '⁴',
            '5': '⁵',
            '6': '⁶',
            '7': '⁷',
            '8': '⁸',
            '9': '⁹',
            '[': '⁽',
            ']': '⁾'
        }
        
        citation = match.group(0)  # Get the number from within the brackets
        return ''.join(superscript_map.get(c, c) for c in citation)

    async def _stream_response(self, headers, payload, citations):
        """
        Handle streaming responses.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=f"{self.valves.PERPLEXITY_API_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
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
                            chunk = json.loads(line)
                            if "citations" in chunk:
                                if isinstance(chunk["citations"], list):
                                    citations.update(chunk["citations"])
                                elif isinstance(chunk["citations"], str):
                                    citations.update(chunk["cutations"])

                            if chunk.get("choices") and chunk["choices"][0].get("delta") and chunk["choices"][0]["delta"].get("content"):
                                content = chunk["choices"][0]["delta"]["content"]

                                modified_content = re.sub(r'\[\d+\]', self._convert_citations_to_superscript, content)

                                if modified_content != content:
                                    logger.debug(f"Converted citations in chunk: {content} -> {modified_content}")
                                    chunk["choices"][0]["delta"]["content"] = modified_content

                                yield chunk

                        except json.JSONDecodeError as je:
                            logger.error(f"JSON decode error for line: {line}")
                            logger.error(f"Error details: {str(je)}")
                            yield f"JSON decode error for line: {line}"
                            yield f"Error details: {str(je)}"
                            continue

                    except UnicodeDecodeError as ue:
                        logger.error(f"Unicode decode error: {str(ue)}")
                        yield f"Unicode decode error: {str(ue)}"
                        continue

                    except Exception as e:
                        logger.exception(f"Unexpected error processing line: {str(e)}")
                        yield f"Unexpected error processing line: {str(e)}"
                        continue

    async def _get_response(self, headers, payload, citations, is_title_gen: bool = False):
        """
        Handle non-streaming responses.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=f"{self.valves.PERPLEXITY_API_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                try:
                    response_json = await response.json()
                    logger.debug(f"response_json: {pformat(response_json)}")
                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        content = response_json["choices"][0]["message"]["content"]
                        logger.debug(f"Accumulated content: {content}")
                    else:
                        logger.error(f"Unexpected response format: {response_json}")
                        return "Error: Unexpected response format from API"

                    if not is_title_gen:
                        if response_json.get("citations") and response_json["citations"] and len(response_json["citations"]) > 0:
                            content = re.sub(r'\[\d+\]', self._convert_citations_to_superscript, content)
                            citations_list = await self._build_citation_list(response_json["citations"])
                            content += "\n\n<details>\n<summary>Sources</summary>\n"
                            for i, citation in enumerate(citations_list, 1):
                                content += f"\n[{i}] [{citation.get('title')}]({citation.get('url')})"
                            content += "\n</details>"

                    return content

                except Exception as e:
                    logger.error(f"Error processing response: {str(e)}")
                    return f"Error: {str(e)}"              

    def pipes(self):
        global logger
        if self.valves.PIPE_DEBUG:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging is enabled for the pipe")
        else:
            logger.setLevel(logging.INFO)
            logger.info("Debug logging is disabled for the pipe")

        return [
            {
                "id": "llama-3.1-sonar-small-128k-online",
                "name": f"{self.valves.NAME_PREFIX}Llama 3.1 Sonar Small 128k Online",
            },
            {
                "id": "llama-3.1-sonar-large-128k-online",
                "name": f"{self.valves.NAME_PREFIX}Llama 3.1 Sonar Large 128k Online",
            },
            {
                "id": "llama-3.1-sonar-huge-128k-online",
                "name": f"{self.valves.NAME_PREFIX}Llama 3.1 Sonar Huge 128k Online",
            },
            {
                "id": "llama-3.1-sonar-small-128k-chat",
                "name": f"{self.valves.NAME_PREFIX}Llama 3.1 Sonar Small 128k Chat",
            },
            {
                "id": "llama-3.1-sonar-large-128k-chat",
                "name": f"{self.valves.NAME_PREFIX}Llama 3.1 Sonar Large 128k Chat",
            },
            {
                "id": "llama-3.1-8b-instruct",
                "name": f"{self.valves.NAME_PREFIX}Llama 3.1 8B Instruct",
            },
            {
                "id": "llama-3.1-70b-instruct",
                "name": f"{self.valves.NAME_PREFIX}Llama 3.1 70B Instruct",
            },
        ]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __metadata__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
    ) -> AsyncGenerator[str, None]:
        logger.debug(f"pipe:{__name__}")

        if not self.valves.PERPLEXITY_API_KEY:
            raise Exception("PERPLEXITY_API_KEY not provided in the valves.")

        headers = {
            "Authorization": f"Bearer {self.valves.PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        user_valves = __user__.get("valves")
        if not user_valves:
            user_valves = self.UserValves()

        citations = set()

        try:
            emitter = EventEmitter(__event_emitter__)

            metadata = await self._build_metadata(__user__, __metadata__, user_valves)
            payload = await self._build_completion_payload(body, __user__, metadata, user_valves, emitter)

            logger.debug(f"Payload: {pformat(payload)}")

            try:
                is_title_gen = __metadata__.get("task") == "title_generation"

                if body["stream"]:
                    logger.debug(f"Streaming response")
                    async for chunk in self._stream_response(headers, payload, citations):
                        yield chunk
                    if citations:
                        await self._process_citations(citations, emitter, is_title_gen)
                else:
                    logger.debug(f"Building response object")
                    content = await self._get_response(headers, payload, citations, is_title_gen)
                    yield content

                await emitter.emit_status(
                    status="complete",
                    description="",
                    done=True,
                )
            except aiohttp.ClientError as e:
                logger.error(f"Error during request: {e}")
                yield f"Error: {e}"
        except Exception as e:
            yield f"Error: {e}"