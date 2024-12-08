"""
title: LiteLLM Manifold Pipe
author: open-webui
date: 2024-05-30
version: 1.0.1
license: MIT
description: A manifold pipe that uses LiteLLM.
requirements: beautifulsoup4, yt_dlp, litellm
"""

from bs4 import BeautifulSoup
from copy import deepcopy
from functools import lru_cache
from pydantic import BaseModel, Field
from typing import Dict, List, AsyncGenerator, Union, Callable, Any, Awaitable
import aiohttp
import json
import litellm
import logging
import requests

from open_webui.utils.misc import get_last_user_message, pop_system_message

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
        REQUEST_TIMEOUT: int = Field(
            default=5,
            description="(Optional) Timeout (in seconds) for aiohttp session requests. Default is 5s."
        )
        YOUTUBE_COOKIES_FILEPATH: str = Field(
            default="",
            description="(Optional) Path to cookies file from youtube.com to aid in title retrieval for citations."
        )
        VISION_ROUTER_ENABLED: bool = Field(
            default=False, description="(Optional) Enable vision rerouting."
        )
        VISION_MODEL_ID: str = Field(
            default="",
            description="(Optional) The identifier of the vision model to be used for processing images."
        )
        SKIP_REROUTE_MODELS: list[str] = Field(
            default_factory=list,
            description="(Optional) A list of model identifiers that should not be re-routed to the chosen vision model.",
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
        pass

    class UserValves(BaseModel):
        EXTRA_METADATA: str = Field(
            default="", description='(Optional) Additional metadata, e.g. {"key": "value"}'
        )
        EXTRA_TAGS: str = Field(
            default='',
            description='(Optional) A list of tags to apply to requests, e.g. ["open-webui"]',
        )


    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        self._model_list = None

    def _get_model_list(self):
        """
        Get a (not open-webui compatible) list of models from LiteLLM, including extra
        properties to help us set provider-specific parameters later.
        """
        logger.debug("Fetching model list from LiteLLM")
        model_list = [
            {'id': 'o1-preview', 'name': 'o1-preview'},
            {'id': 'o1-mini', 'name': 'o1-mini'},
            {'id': 'gpt-4o-mini', 'name': 'gpt-4o-mini'},
            {'id': 'chatgpt-4o-latest', 'name': 'chatgpt-4o-latest'},
            {'id': 'gpt-4o-mini-2024-07-18', 'name': 'gpt-4o-mini-2024-07-18'},
            {'id': 'gpt-4o', 'name': 'gpt-4o'},
            {'id': 'openai/gpt-4o-realtime-preview-2024-10-01', 'name': 'gpt-4o-realtime-preview'},
            {'id': 'gpt-4o-audio-preview-2024-10-01', 'name': 'gpt-4o-audio-preview-2024-10-01'},
            {'id': 'gpt-4o-2024-11-20', 'name': 'gpt-4o-2024-11-20'},
            {'id': 'gpt-4o-2024-08-06', 'name': 'gpt-4o-2024-08-06'},
            {'id': 'gpt-4o-2024-05-13', 'name': 'gpt-4o-2024-05-13'},
            {'id': 'gpt-4-turbo', 'name': 'gpt-4-turbo'},
            {'id': 'gpt-4-0125-preview', 'name': 'gpt-4-0125-preview'},
            {'id': 'gpt-4-1106-preview', 'name': 'gpt-4-1106-preview'},
            {'id': 'gpt-3.5-turbo-1106', 'name': 'gpt-3.5-turbo-1106'},
            {'id': 'gpt-3.5-turbo', 'name': 'gpt-3.5-turbo'},
            {'id': 'gpt-3.5-turbo-16k', 'name': 'gpt-3.5-turbo-16k'},
            {'id': 'gpt-4', 'name': 'gpt-4'},
            {'id': 'gpt-4-0613', 'name': 'gpt-4-0613'},
            {'id': 'text-embedding-3-large', 'name': 'text-embedding-3-large'},
            {'id': 'text-embedding-3-small', 'name': 'text-embedding-3-small'},
            {'id': 'text-embedding-ada-002', 'name': 'text-embedding-ada-002'},
            {'id': 'openai/dall-e-3', 'name': 'dall-e-3'},
            {'id': 'openai/dall-e-2', 'name': 'dall-e-2'},
            {'id': 'whisper-1', 'name': 'whisper-1'},
            {'id': 'tts-1', 'name': 'tts-1'},
            {'id': 'tts-1-hd', 'name': 'tts-1-hd'},
            {'id': 'gpt-3.5-turbo-instruct', 'name': 'gpt-3.5-turbo-instruct'},
            {'id': 'gpt-3.5-turbo-instruct-0914', 'name': 'gpt-3.5-turbo-instruct-0914'},
            {'id': 'text-completion-openai/babbage-002', 'name': 'babbage-002'},
            {'id': 'text-completion-openai/davinci-002', 'name': 'davinci-002'},
            {'id': 'anthropic/claude-3-5-sonnet-20241022', 'name': 'claude-3.5-sonnet'},
            {'id': 'anthropic/claude-3-5-haiku-20241022', 'name': 'claude-3.5-haiku'},
            {'id': 'anthropic/claude-3-sonnet-20240229', 'name': 'claude-3-sonnet'},
            {'id': 'anthropic/claude-3-opus-20240229', 'name': 'claude-3-opus'},
            {'id': 'anthropic/claude-3-haiku-20240307', 'name': 'claude-3-haiku'},
            {'id': 'anthropic/claude-2', 'name': 'claude-2'},
            {'id': 'anthropic/claude-2.1', 'name': 'claude-2.1'},
            {'id': 'gemini/gemini-1.5-flash-002', 'name': 'gemini-1.5-flash-002'},
            {'id': 'gemini/gemini-1.5-flash-001', 'name': 'gemini-1.5-flash-001'},
            {'id': 'gemini/gemini-1.5-flash', 'name': 'gemini-1.5-flash'},
            {'id': 'gemini/gemini-1.5-flash-latest', 'name': 'gemini-1.5-flash-latest'},
            {'id': 'gemini/gemini-1.5-flash-8b-exp-0924', 'name': 'gemini-1.5-flash-8b-exp-0924'},
            {'id': 'gemini/gemini-1.5-flash-exp-0827', 'name': 'gemini-1.5-flash-exp-0827'},
            {'id': 'gemini/gemini-1.5-flash-8b-exp-0827', 'name': 'gemini-1.5-flash-8b-exp-0827'},
            {'id': 'gemini/gemini-pro', 'name': 'gemini-pro'},
            {'id': 'gemini/gemini-1.5-pro', 'name': 'gemini-1.5-pro'},
            {'id': 'gemini/gemini-1.5-pro-002', 'name': 'gemini-1.5-pro-002'},
            {'id': 'gemini/gemini-1.5-pro-001', 'name': 'gemini-1.5-pro-001'},
            {'id': 'gemini/gemini-1.5-pro-exp-0801', 'name': 'gemini-1.5-pro-exp-0801'},
            {'id': 'gemini/gemini-1.5-pro-exp-0827', 'name': 'gemini-1.5-pro-exp-0827'},
            {'id': 'gemini/gemini-1.5-pro-latest', 'name': 'gemini-1.5-pro-latest'},
            {'id': 'gemini/gemini-exp-1114', 'name': 'gemini-exp-1114'},
            {'id': 'gemini/gemini-exp-1206', 'name': 'gemini-exp-1206'},
            {'id': 'codestral/codestral-latest', 'name': 'codestral-latest'},
            {'id': 'codestral/codestral-2405', 'name': 'codestral-2405'},
            {'id': 'deepseek/deepseek-chat', 'name': 'deepseek-chat'},
            {'id': 'deepseek/deepseek-coder', 'name': 'deepseek-coder'},
            {'id': 'mistral/open-mistral-7b', 'name': 'open-mistral-7b'},
            {'id': 'mistral/open-mistral-nemo', 'name': 'open-mistral-nemo'},
            {'id': 'mistral/open-mistral-nemo-2407', 'name': 'open-mistral-nemo-2407'},
            {'id': 'mistral/open-mixtral-8x7b', 'name': 'open-mixtral-8x7b'},
            {'id': 'mistral/open-mixtral-8x22b', 'name': 'open-mixtral-8x22b'},
            {'id': 'mistral/mistral-embed', 'name': 'mistral-embed'},
            {'id': 'mistral/open-codestral-mamba', 'name': 'open-codestral-mamba'},
            {'id': 'mistral/codestral-mamba-latest', 'name': 'codestral-mamba-latest'},
            {'id': 'voyage/voyage-multimodal-3', 'name': 'voyage-multimodal-3'},
            {'id': 'voyage/voyage-3', 'name': 'voyage-3'},
            {'id': 'voyage/voyage-3-lite', 'name': 'voyage-3-lite'},
            {'id': 'voyage/voyage-2', 'name': 'voyage-2'},
            {'id': 'voyage/voyage-finance-2', 'name': 'voyage-finance-2'},
            {'id': 'voyage/voyage-law-2', 'name': 'voyage-law-2'},
            {'id': 'voyage/voyage-code-3', 'name': 'voyage-code-3'},
            {'id': 'voyage/voyage-code-2', 'name': 'voyage-code-2'},
            {'id': 'voyage/rerank-2', 'name': 'rerank-2'},
            {'id': 'voyage/rerank-2-lite', 'name': 'rerank-2-lite'},
            {'id': 'openai/nvdev/nvidia/llama-3.1-nemotron-70b-instruct', 'name': 'nvdev-llama-3.1-nemotron-70b-instruct'},
            {'id': 'openai/nvdev/nvidia/llama-3.1-70b-instruct', 'name': 'nvdev-llama-3.1-70b-instruct'},
            {'id': 'openai/nvdev/meta/llama-3.2-90b-vision-instruct', 'name': 'nvdev-meta-llama-3.2-90b-vision-instruct'},
            {'id': 'openai/nvdev/mistralai/mixtral-8x22b-instruct-v0.1', 'name': 'nvdev-mistralai-mixtral-8x22b-instruct-v0.1'},
            {'id': 'openai/nvdev/nvidia/embed-qa-4', 'name': 'nvdev-nvidia-embed-qa-4'},
            {'id': 'openai/nvdev/nvidia/nv-embedqa-e5-v5', 'name': 'nvdev-nvidia-nv-embedqa-e5-v5'},
            {'id': 'openai/nvdev/nvidia/nv-rerankqa-mistral-4b-v3', 'name': 'nvdev-nvidia-nv-rerankqa-mistral-4b-v3'},
            {'id': 'openai/nvdev/meta/llama-3.2-3b-instruct', 'name': 'nvdev-meta-llama-3.2-3b-instruct'},
            {'id': 'openai/nvdev/meta/llama-3.2-1b-instruct', 'name': 'nvdev-meta-llama-3.2-1b-instruct'},
            {'id': 'openai/nvdev/bigcode/starcoder2-7b', 'name': 'nvdev-bigcode-starcoder2-7b'},
            {'id': 'openai/nvdev/nvidia/nemotron-mini-4b-instruct', 'name': 'nvdev-nvidia-nemotron-mini-4b-instruct'},
            {'id': 'bedrock/stability.sd3-large-v1:0', 'name': 'bedrock-stability-sd3-large-v1'},
            {'id': 'cohere/command-r-plus-08-2024', 'name': 'command-r-plus-08-2024'},
            {'id': 'cohere/command-r-08-2024', 'name': 'command-r-08-2024'},
            {'id': 'cohere/command-r-plus', 'name': 'command-r-plus'},
            {'id': 'cohere/command-r', 'name': 'command-r'},
            {'id': 'cohere/command-r-light', 'name': 'command-r-light'},
            {'id': 'cohere/rerank-english-v3.0', 'name': 'rerank-english-v3.0'},
            {'id': 'cohere/embed-english-v3.0', 'name': 'embed-english-v3.0'},
            {'id': 'cohere/embed-english-light-v3.0', 'name': 'embed-english-light-v3.0'},
            {'id': 'cohere/embed-multilingual-v3.0', 'name': 'embed-multilingual-v3.0'},
            {'id': 'cohere/embed-multilingual-light-v3.0', 'name': 'embed-multilingual-light-v3.0'},
            {'id': 'perplexity/llama-3.1-sonar-small-128k-online', 'name': 'llama-3.1-sonar-small-128k-online'},
            {'id': 'perplexity/llama-3.1-sonar-large-128k-online', 'name': 'llama-3.1-sonar-large-128k-online'},
            {'id': 'perplexity/llama-3.1-sonar-huge-128k-online', 'name': 'llama-3.1-sonar-huge-128k-online'},
            {'id': 'deepinfra/meta-llama/Llama-3.3-70B-Instruct', 'name': 'meta-llama/Llama-3.3-70B-Instruct'},
            {'id': 'deepinfra/Qwen/Qwen2.5-72B-Instruct-Turbo', 'name': 'Qwen/Qwen2.5-72B-Instruct-Turbo'},
            {'id': 'deepinfra/openai/whisper-large-v3', 'name': 'openai/whisper-large-v3'},
            {'id': 'deepinfra/stabilityai/sd3.5', 'name': 'stabilityai-sd3.5'},
            {'id': 'together_ai/black-forest-labs/FLUX.1-dev', 'name': 'black-forest-labs/FLUX.1-dev'},
            {'id': 'together_ai/black-forest-labs/FLUX.1-canny', 'name': 'black-forest-labs/FLUX.1-canny'},
            {'id': 'together_ai/black-forest-labs/FLUX.1-schnell', 'name': 'black-forest-labs/FLUX.1-schnell'},
            {'id': 'together_ai/black-forest-labs/FLUX.1-depth', 'name': 'black-forest-labs/FLUX.1-depth'},
            {'id': 'together_ai/black-forest-labs/FLUX.1-redux', 'name': 'black-forest-labs/FLUX.1-redux'},
            {'id': 'together_ai/black-forest-labs/FLUX.1-pro', 'name': 'black-forest-labs/FLUX.1-pro'},
            {'id': 'together_ai/black-forest-labs/FLUX.1.1-pro', 'name': 'black-forest-labs/FLUX.1.1-pro'},
            {'id': 'together_ai/stabilityai/stable-diffusion-xl-base-1.0', 'name': 'stabilityai/stable-diffusion-xl-base-1.0'},
            {'id': 'ollama_chat/llama3.2:latest', 'name': 'ollama-llama-3.2-8b'},
            {'id': 'ollama_chat/llama-3.2-3b-instruct-f16-uncensored:latest', 'name': 'ollama-llama-3.2-3b-instruct-f16-uncensored'},
            {'id': 'ollama_chat/llama-3.2-3b-instruct-abliterated:latest', 'name': 'ollama-llama-3.2-3b-instruct-abliterated'},
            {'id': 'ollama_chat/dolphin-mistral:latest', 'name': 'ollama-dolphin-mistral'},
            {'id': 'ollama_chat/gemma2:27b', 'name': 'ollama-gemma2-27b'},
            {'id': 'ollama_chat/marco/em_german_mistral_v01:latest', 'name': 'ollama-marco-em-german-mistral-v01'},
        ]
        return model_list

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
        and any provider-specific parameters, using the model list from _get_model_list()
        to identify the provider of the model we're talking to.
        """
        # Required parameters
        logger.debug(f"{body['model']}")
        model_parts = body["model"].split(".", 1) # models from this pipe look like e.g. "litellm_native.gpt-4o" or "litellm_native.anthropic/claude-3-5-sonnet"
        litellm_prefix = model_parts[0] if len(model_parts) > 1 else "" # "litellm_native"

        model_fullname = model_parts[1] if len(model_parts) > 1 else body["model"] # "gpt-4o", "anthropic/claude-3-5-sonnet"
        model_fullname_parts = model_fullname.split(".", 1)
        provider = model_fullname_parts[0] if len(model_fullname_parts) > 1 else "" # "", "anthropic"
        model_name = model_fullname_parts[1] if len(model_fullname_parts) > 1 else model_fullname # "gpt-4o", "claude-3-5-sonnet"

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
            logger.debug(f"CURRENT MODEL NAME: {model_fullname}")

            if self.valves.VISION_MODEL_ID:
                # If we're not already using the rerouted models and we're not using a model that we want to skip rerouting in, reroute it
                if model_fullname != self.valves.VISION_MODEL_ID and model_fullname not in self.valves.SKIP_REROUTE_MODELS:
                    model_fullname = self.valves.VISION_MODEL_ID
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
            if m["name"] == model_fullname or m["id"] == model_fullname:
                logger.debug(f"Matched {m['name']} (model id: {m['id']}) with {model_fullname}")
                if "provider" in m and m["provider"]:
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
            "model": model_fullname, # optional vision model if image in last message
            "messages": [{"role": "system", "content": system_prompt}, *cleaned_messages], # only most recent message contains data blobs
            # "stream": True,
            "drop_params": True, # litellm
        }

        # Only add openai parameters that differ from defaults
        for param in optional_openai_params:
            if param in body:
                payload[param] = body[param]

        # Add provider-specific parameters if applicable
        if provider and provider in provider_params:
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

    async def _stream_response(self, payload, citations):
        """
        Handle streaming responses.
        """
        response = await litellm.acompletion(
            stream=True,
            **payload
        )
        async for chunk in response:
            if "citations" in chunk:
                if isinstance(chunk["citations"], list):
                    citations.update(chunk["citations"])
                elif isinstance(chunk["citations"], str):
                    citations.add(chunk["citations"])
            yield chunk

    async def _get_response(self, payload, citations):
        """
        Handle non-streaming responses.
        """
        response = await litellm.acompletion(
            stream=False,
            **payload
        )

        logger.debug(f"Accumulated content: {response}")

        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, 'message') and choice.message:
                if hasattr(choice.message, 'content'):
                    message_content = choice.message.content
                    return message_content
                else:
                    logger.error("Message object is missing the 'content' attribute")
            else:
                logger.error("Choice object is missing the 'message' attribute or message is empty")
        else:
            logger.error("The 'choices' list is empty")

        return response.choices[0].message.content

    def pipes(self):
        """
        Get the list of models and return it to Open-WebUI.
        """

        #Set the logging level according to the valve preference.
        global logger
        if self.valves.LITELLM_PIPELINE_DEBUG:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging is enabled for LiteLLM pipe")
        else:
            logger.setLevel(logging.INFO)
            logger.info("Debug logging is disabled for LiteLLM pipe")

        logger.debug("pipes() called - fetching model list")
        return self._get_model_list()

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

        user_valves = __user__.get("valves")
        if not user_valves:
            user_valves = self.UserValves()

        citations = set()

        if self._model_list is None:
            logger.warning("Model list not initialized - this shouldn't happen")
            self._model_list = self._get_model_list()

        try:
            emitter = EventEmitter(__event_emitter__)

            metadata = await self._build_metadata(__user__, __metadata__, user_valves)
            payload = await self._build_completion_payload(body, __user__, metadata, user_valves, emitter)

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

            try:
                is_title_gen = __metadata__.get("task") == "title_generation"

                if body["stream"]:
                    async for chunk in self._stream_response(payload, citations):
                        yield chunk
                else:
                    content = await self._get_response(payload, citations)
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