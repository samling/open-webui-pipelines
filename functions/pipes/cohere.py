"""
title: Cohere Manifold Pipe
author: samling, based on pipe by justinh-rahb and moblangeois
author_url: https://github.com/samling/open-webui-pipelines
version: 0.1.0
license: MIT
requirements: cohere
"""

from functools import lru_cache
from openai import AsyncOpenAI
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
import cohere
import json
import logging
import os
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
            default="Cohere.",
            description="The prefix applied before the model names.",
        )
        BASE_URL: str = Field(
            default="https://api.cohere.com/v2",
            description="The base URL for OpenAI-compatible API endpoint.",
        )
        API_KEY: str = Field(
            default="",
            description="Required API key to access OpenAI-compatible API services.",
        )
        PIPE_DEBUG: bool = Field(
            default=False, description="(Optional) Enable debugging for the pipe."
        )
        EXTRA_METADATA: str = Field(
            default="", description='(Optional) Additional metadata, e.g. {"key": "value"}'
        )

    class UserValves(BaseModel):
        EXTRA_METADATA: str = Field(
            default="", description='(Optional) Additional metadata, e.g. {"key": "value"}'
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

    def _get_openai_models(self):
            if self.valves.API_KEY:
                try:
                    headers = {}
                    headers["Authorization"] = f"bearer {self.valves.API_KEY}"

                    r = requests.get(
                        f"{self.valves.BASE_URL}/models", headers=headers
                    )

                    models = r.json()
                    return [
                        {
                            "id": model["name"],
                            "name": f"{self.valves.NAME_PREFIX}{model['name']}" if 'name' in model else f"{self.valves.NAME_PREFIX}{model['id']}",
                        }
                        for model in models["models"]
                        if ("chat" in model["endpoints"])
                    ]

                except Exception as e:
                    logger.debug(f"Error: {e}")
                    return [
                        {
                            "id": "error",
                            "name": "Could not fetch models from Cohere, please update the API Key in the valves.",
                        },
                    ]
            else:
                return []

    async def _build_metadata(self, __user__, __metadata__, user_valves):
        """
        Construct additional metadata to add to the request.
        This includes trace data to be sent to an observation platform like Langfuse.
        """
        metadata = {}

        extra_metadata = load_json(self.valves.EXTRA_METADATA)
        __metadata__.update(extra_metadata)

        logger.debug(f"User valves: {user_valves}")
        logger.debug(f"User metadata: {user_valves.EXTRA_METADATA}")
        extra_user_metadata = load_json(user_valves.EXTRA_METADATA)
        __metadata__.update(extra_user_metadata)

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

        # Get all messages
        messages = body.get("messages", [])
        # Process system prompt if there is one
        system_message, messages = pop_system_message(messages)
        system_prompt = "You are a helpful assistant."
        if system_message is not None:
            logger.debug(f"Using non-default system prompt: {system_message['content']}")
            system_prompt = system_message["content"]

        # Clean base64-encoded images from previous messages
        logger.debug(f"Stripping encoded image data from past messages")
        cleaned_messages = []
        last_user_message_content = get_last_user_message(messages)
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
        logger.debug(f"Cleaned messages:\n\t{pformat(cleaned_messages)}")

        # Get most recent user message and format it for Cohere
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
        }

        # Only add openai parameters that differ from defaults
        for param in optional_openai_params:
            if param in body:
                payload[param] = body[param]

        # Add metadata if it exists
        if metadata:
            payload["metadata"] = metadata

        logger.debug(f"Built payload with {len(payload)} parameters")

        return payload

    async def _stream_response(self, payload):
        """
        Handle streaming responses.
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.valves.API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            r = requests.post(
                url=f"{self.valves.BASE_URL}/chat",
                json=payload,
                headers=headers,
                stream=True,
            )
            r.raise_for_status()
            logger.debug(r.text)

            for line in r.iter_lines():
                if line:
                    try:
                        event = json.loads(line.decode("utf-8"))
                        if event["type"] == "content-start":
                            logger.debug(f"Received start event: {event}")
                        elif event["type"] == "content-delta":
                            yield event["delta"]["message"]["content"]["text"]
                        #elif event["type"] == "content-end": # TODO: Do we need this vs. message-end?
                        elif event["type"] == "message-end":
                            break
                    except json.JSONDecodeError:
                        logger.debug(f"Failed to decode JSON: {line}")
                        pass
        except requests.RequestException as e:
            print(f"Request exception in stream_response: {e}")
            print(
                f"Response content: {r.content if 'r' in locals() else 'No response'}"
            )
            yield f"Error: {str(e)}"

    async def _get_response(self, payload, is_title_gen: bool = False):
        """
        Handle non-streaming responses.
        """
        headers = {
            "Authorization": f"Bearer {self.valves.API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            r = requests.post(
                url=f"{self.valves.BASE_URL}/chat",
                json=payload,
                headers=headers,
            )
            r.raise_for_status()
            data = r.json()
            logger.debug(f"Completion response: {json.dumps(data, indent=2)}")
            if data.get("message") and data["message"] is not None:
                message = data["message"]
                if message.get("content") and message["content"] is not None:
                    content = message["content"]
                    response = content[0]["text"]
            return response

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

        return self._get_openai_models()

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __metadata__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
    ) -> AsyncGenerator[str, None]:
        logger.debug(f"pipe:{__name__}")

        if not self.valves.API_KEY:
            raise Exception("API_KEY not provided in the valves.")

        user_valves = __user__.get("valves")
        if not user_valves:
            user_valves = self.UserValves()

        try:
            emitter = EventEmitter(__event_emitter__)

            metadata = await self._build_metadata(__user__, __metadata__, user_valves)
            payload = await self._build_completion_payload(body, __user__, metadata, user_valves, emitter)

            logger.debug(f"Payload: {pformat(payload)}")

            try:
                is_title_gen = __metadata__.get("task") == "title_generation"

                if body["stream"]:
                    logger.debug(f"Streaming response")
                    async for chunk in self._stream_response(payload):
                        yield chunk
                else:
                    logger.debug(f"Building response object")
                    content = await self._get_response(payload, is_title_gen)
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