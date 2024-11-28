"""
title: LiteLLM Manifold Pipeline
author: open-webui
date: 2024-05-30
version: 1.0.1
license: MIT
description: A manifold pipeline that uses LiteLLM.
requirements: litellm, langfuse
"""

from functools import cache
from langfuse.decorators import langfuse_context, observe
from typing import List, Union, Generator, Iterator, AsyncGenerator
from pprint import pformat
from pydantic import BaseModel, Field
import asyncio
import litellm
from litellm.integrations.custom_logger import CustomLogger
import json
import os
import requests

@cache
def load_json_dict(user_value: str) -> dict:
    user_value = user_value.strip()
    if not user_value:
        return {}
    loaded = json.loads(user_value)
    assert isinstance(loaded, dict), f"json is not a dict but '{type(loaded)}'"
    return loaded

@cache
def load_json_list(user_value: str) -> list:
    user_value = user_value.strip()
    if not user_value:
        return []
    loaded = json.loads(user_value)
    assert isinstance(loaded, list), f"json is not a list but '{type(loaded)}'"
    assert all(isinstance(elem, str) for elem in loaded), f"List contained non strings elements: '{loaded}'"
    return loaded


class LogHandler(CustomLogger):
    #### ASYNC #### 
    
    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Async Streaming")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(kwargs["litellm_params"]["metadata"])
        print(f"On Async Success")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Async Failure")

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
            default=True,
            description="Enable debugging."
        )
        LANGFUSE_PUBLIC_KEY: str = Field(
            default="pk-fake-key",
            description="Public key for Langfuse.",
        )
        LANGFUSE_PRIVATE_KEY: str = Field(
            default="sk-fake-key",
            description="Private key for Langfuse.",
        )
        LANGFUSE_HOST: str = Field(
            default="http://langfuse.langfuse:3000",
            description="Base URL for Langfuse.",
        )
        pass

    def __init__(self):
        # You can also set the pipelines that are available in this pipeline.
        # Set manifold to True if you want to use this pipeline as a manifold.
        # Manifold pipelines can have multiple pipelines.
        self.type = "manifold"

        # self.id = "litellm_manifold"

        # Optionally, you can set the name of the manifold pipeline.
        # self.name = "Async: "

        # Initialize rate limits
        self.valves = self.Valves()

        self.debug_prefix = "DEBUG:    " + __name__ + " -"
        pass

    def pipes(self):

        headers = {"Content-Type": "application/json"}
        if self.valves.LITELLM_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.LITELLM_API_KEY}"

        os.environ["LANGFUSE_PUBLIC_KEY"] = self.valves.LANGFUSE_PUBLIC_KEY
        os.environ["LANGFUSE_PRIVATE_KEY"] = self.valves.LANGFUSE_PRIVATE_KEY
        os.environ["LANGFUSE_HOST"] = self.valves.LANGFUSE_HOST

        if self.valves.LITELLM_BASE_URL:
            try:
                r = requests.get(
                    f"{self.valves.LITELLM_BASE_URL}/v1/models", headers=headers
                )
                models = r.json()
                return [
                    {
                        "id": model["id"],
                        "name": model["name"] if "name" in model else model["id"],
                    }
                    for model in models["data"]
                ]
            except Exception as e:
                print(f"Error fetching models from LiteLLM: {e}")
                return [
                    {
                        "id": "error",
                        "name": "Could not fetch models from LiteLLM, please update the URL in the valves.",
                    },
                ]
        else:
            print("LITELLM_BASE_URL not set. Please configure it in the valves.")
            return []

    async def _stream_response(self, response):
        async for line in response.content:
            if line:
                line = line.decode('utf-8')
                if not line or line == "data: ":
                    continue

                if line.startswith("data: "):
                    line = line[6:]
                
                try:
                    json_data = json.loads(line)
                    if "choices" in json_data and len(json_data["choices"]):
                        delta = json_data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield json_data
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON: {line}")
                    continue

    async def _get_response(self, response):
        return await response.json()

    logHandler = LogHandler()
    litellm.callbacks = [logHandler]

    @observe()
    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __metadata__: dict
    ) -> AsyncGenerator[str, None]:
        if self.valves.LITELLM_PIPELINE_DEBUG:
            print(f"{self.debug_prefix} Metadata: {__metadata__}")
            print(f"{self.debug_prefix} Body: {body}")

        try:
            curr_trace_id = langfuse_context.get_current_trace_id()
            print(f"{self.debug_prefix} Trace ID: {curr_trace_id}")

            model = body["model"].split(".")[-1]
            metadata = {
                "tags": ["open-webui"],
                "chat_id": __metadata__.get("chat_id"),
                "user_id": __user__.get("id"),
                "user_name": __user__.get("name"),
            }
            response = await litellm.acompletion(
                model=model,
                messages=body.get("messages", []),
                stream=body.get("stream", True),
                api_key=self.valves.LITELLM_API_KEY,
                base_url=self.valves.LITELLM_BASE_URL,
                max_tokens=body.get("max_tokens", None),
                user=__user__.get("name"),
                metadata=metadata
            )

            if body.get("stream", True):
                async for chunk in response:
                    yield chunk
            else:
                content = response.choices[0].message.content
                yield content

        except Exception as e:
            print(f"Error during request: {e}")
            yield {"error": str(e)}