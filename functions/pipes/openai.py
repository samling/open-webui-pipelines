"""
title: OpenAI Manifold Pipe
author: samling, based on pipe by justinh-rahb and moblangeois
author_url: https://github.com/samling/open-webui-pipelines
version: 0.1.0
license: MIT
"""

from functools import lru_cache
from openai import OpenAI
from pprint import pformat
from pydantic import BaseModel, Field
from typing import (
    Any,
    Generator,
    Callable,
    Dict,
    List,
    Union,
)
from utils.misc import get_last_user_message
from utils.misc import pop_system_message

import json
import logging
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

class Pipe:
    class Valves(BaseModel):
        NAME_PREFIX: str = Field(
            default="OpenAI.",
            description="The prefix applied before the model names.",
        )
        BASE_URL: str = Field(
            default="https://api.openai.com/v1",
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

    def _get_openai_models(self, custom_models=None):
            if custom_models is None:
                custom_models = []

            if self.valves.API_KEY:
                try:
                    headers = {}
                    headers["Authorization"] = f"Bearer {self.valves.API_KEY}"
                    headers["Content-Type"] = "application/json"

                    r = requests.get(
                        f"{self.valves.BASE_URL}/models", headers=headers
                    )

                    models = r.json()
                    base_models = [
                        {
                            "id": model["id"],
                            "name": f"{self.valves.NAME_PREFIX}{model['name']}" if 'name' in model else f"{self.valves.NAME_PREFIX}{model['id']}",
                        }
                        for model in models["data"]
                        if ("gpt" in model["id"]) or ("o1" in model["id"])
                    ]

                except Exception as e:

                    print(f"Error: {e}")
                    return [
                        {
                            "id": "error",
                            "name": "Could not fetch models from OpenAI, please update the API Key in the valves.",
                        },
                    ]
            for model in custom_models:
                if not model["name"].startswith(self.valves.NAME_PREFIX):
                    model["name"] = f"{self.valves.NAME_PREFIX}{model['name']}"

            return base_models + custom_models

    def _build_metadata(self, __user__, __metadata__, user_valves):
        metadata = {}

        extra_metadata = load_json(self.valves.EXTRA_METADATA)
        __metadata__.update(extra_metadata)

        logger.debug(f"User valves: {user_valves}")
        logger.debug(f"User metadata: {user_valves.EXTRA_METADATA}")
        extra_user_metadata = load_json(user_valves.EXTRA_METADATA)
        __metadata__.update(extra_user_metadata)

        return metadata

    def _build_completion_payload(
        self,
        body: dict,
        __user__: dict,
        metadata: dict,
        user_valves: UserValves,
    ) -> dict:
        logger.debug(f"Model from open-webui request: {body['model']}")
        manfold_name, model_name = self._parse_model_string(body['model'])

        messages = body.get("messages", [])
        last_user_message_content = get_last_user_message(messages)
        if last_user_message_content is None:
            return body

        system_message, messages = pop_system_message(messages)
        system_prompt = "You are a helpful assistant."
        if system_message is not None:
            logger.debug(f"Using non-default system prompt: {system_message['content']}")
            system_prompt = system_message["content"]

        logger.debug(f"Stripping encoded image data from past messages")
        cleaned_messages = []
        for message in messages:
            cleaned_message = message.copy()
            if isinstance(message.get("content"), list):
                if (message["role"] == "user" and 
                    any(item.get("type") == "text" and item.get("text") == last_user_message_content
                        for item in message["content"])):
                    cleaned_message = message
                else:
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

        optional_openai_params = {
            "seed",
            "temperature",
            "top_p",
            "max_tokens",
            "frequency_penalty",
            "stop",
        }

        payload = {
            "model": model_name,
            "messages": [{"role": "system", "content": system_prompt}, *cleaned_messages],
            "stream": body.get("stream", True),
        }

        for param in optional_openai_params:
            if param in body:
                payload[param] = body[param]

        if __user__.get("id"):
            payload["user"] = __user__["id"]

        if metadata:
            payload["metadata"] = metadata

        logger.debug(f"Built payload with {len(payload)} parameters")

        return payload

    def _stream_response(self, client: OpenAI, payload):
        try:
            stream = client.chat.completions.create(
                **payload
            )
            for chunk in stream:
                if chunk and chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content is not None:
                        chunk_dict = {
                            "choices": [{
                                "delta": {
                                    "content": delta.content
                                }
                            }]
                        }
                        yield chunk_dict
        except Exception as e:
            logger.error(f"Error details: {str(e)}")
            yield f"Error details: {str(e)}"

    def _get_response(self, client: OpenAI, payload, is_title_gen: bool = False):
        try:
            response = client.chat.completions.create(**payload)
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                logger.debug(f"Accumulated content: {content}")
                return content
            else:
                logger.error(f"Unexpected response format: {response}")
                return "Error: Unexpected response format from API"

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

        custom_models = [
            {
                "id": "o1",
                "name": "o1"
            },
            {
                "id": "o1-2024-12-17", 
                "name": "o1-2024-12-17"
            }
        ]

        return self._get_openai_models(custom_models)

    def pipe(
        self,
        body: dict,
        __user__: dict,
        __metadata__: dict,
    ) -> Generator[str, None, None]:
        logger.debug(f"pipe:{__name__}")

        if not self.valves.API_KEY:
            raise Exception("API_KEY not provided in the valves.")

        user_valves = __user__.get("valves")
        if not user_valves:
            user_valves = self.UserValves()

        client = OpenAI(
            base_url=self.valves.BASE_URL,
            api_key=self.valves.API_KEY
        )

        try:
            metadata = self._build_metadata(__user__, __metadata__, user_valves)
            payload = self._build_completion_payload(body, __user__, metadata, user_valves)

            logger.debug(f"Payload: {pformat(payload)}")

            try:
                is_title_gen = __metadata__.get("task") == "title_generation"

                if body["stream"]:
                    logger.debug(f"Streaming response")
                    for chunk in self._stream_response(client, payload):
                        yield chunk
                else:
                    logger.debug(f"Building response object")
                    content = self._get_response(client, payload, is_title_gen)
                    return content

            except Exception as e:
                logger.error(f"Error during request: {e}")
                yield f"Error: {e}"
        except Exception as e:
            yield f"Error: {e}"