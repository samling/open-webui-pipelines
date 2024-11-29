"""
title: LiteLLM Manifold Pipe
author: open-webui
date: 2024-05-30
version: 1.0.1
license: MIT
description: A manifold pipe that uses LiteLLM.
"""

from typing import Dict, List, AsyncGenerator, Union, Callable, Any, Awaitable
from pprint import pformat
from pydantic import BaseModel, Field
from functools import lru_cache
import aiohttp
import json
import logging
import os
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
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

class Pipe:
    class Valves(BaseModel):
        LITELLM_BASE_URL: str = Field(
            default="http://litellm.litellm:4000",
            description="Base URL for LiteLLM.",
        )
        LITELLM_API_KEY: str = Field(
            default="sk-fake-key",
            description="Your LiteLLM master key.",
        )
        LITELLM_PIPELINE_DEBUG: bool = Field(
            default=False, description="Enable debugging."
        )
        EXTRA_METADATA: str = Field(
            default="", description='Additional metadata, e.g. {"key": "value"}'
        )
        EXTRA_TAGS: str = Field(
            default='["open-webui"]',
            description='A list of tags to apply to requests, e.g. ["open-webui"]',
        )
        pass

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.current_citations = set()

        if self.valves.LITELLM_PIPELINE_DEBUG:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging is enabled for LiteLLM pipe")

    def _enable_debug(self):
        if not logger.isEnabledFor(logging.DEBUG):
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging is enabled for LiteLLM pipe")

    async def _stream_response(self, response):
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
        logger.debug(f"Response status: {response.status}")
        logger.debug(f"Response headers: {response.headers}")

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

    def pipes(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.LITELLM_API_KEY}",
        }

        try:
            r = requests.get(
                f"{self.valves.LITELLM_BASE_URL}/v1/models", headers=headers
            )
            r.raise_for_status()
            models = r.json()


            model_list = [
                {
                    "id": model["id"],
                    "name": model["name"] if "name" in model else model["id"],
                }
                for model in models.get("data", [])
            ]

            # for model in model_list:
            #     logger.debug(f"Processed model - id: {model['id']}, name: {model['name']}")
            
            return model_list

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching models from LiteLLM: {e}")
            return []

    async def _build_metadata(self, __user__, __metadata__):
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
        # Required parameters
        model_parts = body["model"].split(".", 1)
        provider = model_parts[0] if len(model_parts) > 1 else None
        model = model_parts[1] if len(model_parts) > 1 else body["model"]

        logger.debug(f"Model provider: {provider}")
        logger.debug(f"Model name: {model}")

        payload = {
            "model": model,
            "messages": body.get("messages", []),
            "stream": True,
            "drop_params": True
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

        # Add user and metadata if they exist
        if __user__.get("id"):
            payload["user"] = __user__["id"]

        if metadata:
            payload["metadata"] = metadata

        logger.debug(f"Built payload with {len(payload)} parameters")

        return payload

    async def pipe(
        self, body: dict, __user__: dict, __metadata__: dict, __event_emitter__: Callable[[Any], Awaitable[None]],
    ) -> AsyncGenerator[str, None]:

        if self.valves.LITELLM_PIPELINE_DEBUG:
            self._enable_debug()
            logger.debug("Starting pipe execution")
            logger.debug(f"Metadata: {__metadata__}")
            logger.debug(f"Body: {body}")

        headers = {"Content-Type": "application/json"}
        if self.valves.LITELLM_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.LITELLM_API_KEY}"

        try:
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

                        if body["stream"]:
                            async for chunk in self._stream_response(response):
                                yield chunk
                        else:
                            content = await self._get_response(response)
                            yield content

                        for i, url in enumerate(list(self.current_citations)):
                            await __event_emitter__(
                                {
                                    "type": "citation",
                                    "data": {
                                        "source": {"name": url},
                                        "document": [url],
                                        "metadata": [{"source": url}],
                                    },
                                },
                            )

                    except aiohttp.ClientError as e:
                        logger.error(f"Error during request: {e}")
                        yield f"Error: {e}"
        except Exception as e:
            logger.exception(f"Unexpected error in pipe: {e}")
            print(f"Error: {e}")
