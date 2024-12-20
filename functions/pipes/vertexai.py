"""
title: Google GenAI (Vertex AI) Manifold Pipeline
author: Sam B
date: 2024-12-18
version: 0.1.0
license: MIT
description: A pipeline for generating text using Google's GenAI models in Open-WebUI.
requirements: google
"""

import base64
import google
import google.genai
import logging

from google.genai.types import Tool, GenerateContentConfig, GoogleSearch, GoogleSearchRetrieval
from google.genai.types import (
    Content,
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
from open_webui.utils.misc import get_last_user_message, pop_system_message
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def emit_message(self, content=""):
        if self.event_emitter:
            await self.event_emitter({"type": "message", "data": {"content": content}})

    async def emit_status(
        self, description="Unknown State", status="in_progress", done=False
    ):
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
                        "metadata": [{"source": name, "html": html}],
                        "source": {"name": name, "url": url},
                    },
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
                        "source": source,
                    },
                }
            )

class Pipe:
    class Valves(BaseModel):
        """Options to change from the WebUI"""
        NAME_PREFIX: str = Field(
            default="VertexAI.",
            description="The prefix applied before the model names.",
        )
        GOOGLE_API_KEY: str = Field(
            default="fake-key",
            description="Google API key."
        )
        USE_PERMISSIVE_SAFETY: bool = Field(
            default=False,
            description="Globally enable permissive safety."
        )
        ENABLE_GROUNDING: bool = Field(
            default=False,
            description="Globally enable web search-based grounding."
        )
        PIPE_DEBUG: bool = Field(
            default=False,
            description="Enable pipe debugging."
        )

    class UserValves(BaseModel):
        USE_PERMISSIVE_SAFETY: bool = Field(
            default=False,
            description="Enable permissive safety."
        )
        ENABLE_GROUNDING: bool = Field(
            default=False,
            description="Enable web search-based grounding."
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.client: google.genai.Client | None = None

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

    def _build_conversation_history(self, messages: List[dict]) -> List[Content]:
        contents = []

        def debug_print_contents(contents: List[Content]):
            """Helper function to print contents in a familiar LLM format"""
            debug_list = []
            for content in contents:
                message = {"role": content.role}
                # Combine all text parts into a single text field
                text_parts = [p.text for p in content.parts if p.text]
                if text_parts:
                    message["text"] = " ".join(text_parts)
                # Note image/binary parts
                binary_parts = [p for p in content.parts if p.inline_data or p.file_data]
                if binary_parts:
                    message["has_image"] = True
                debug_list.append(message)
            
            logger.debug(f"Conversation history:\n{pformat(debug_list)}")

        def create_text_only_parts(message, include_placeholders=True) -> List[Part]:
            """Extract only text content from a message"""
            if isinstance(message.get("content"), list):
                text_parts = []
                for content in message["content"]:
                    if content["type"] == "text":
                        text_parts.append(Part.from_text(content["text"]))
                    elif content["type"] == "image_url" and include_placeholders:
                        parts.append(Part.from_text("[binary_data]"))
                return text_parts if text_parts else [Part.from_text("")]
            else:
                return [Part.from_text(message["content"])]

        for message in messages:
            if message["role"] == "system":
                continue

            is_user = message["role"] == "user"
            is_current_message = message == messages[-1]

            if is_user and is_current_message:
                parts = []
                if isinstance(message.get("content"), list):
                    for content in message["content"]:
                        if content["type"] == "text":
                            parts.append(Part.from_text(content["text"]))
                        elif content["type"] == "image_url" and is_user: # skip images from model responses
                            image_url = content["image_url"]["url"]
                            if image_url.startswith("data:image"):
                                metadata, base64_data = image_url.split(",", 1)
                                mime_type = metadata.split(";")[0].split(":")[1]
                                image_bytes = base64.b64decode(base64_data)
                                logger.debug(f"Appending image with mime_type {mime_type}")
                                parts.append(Part.from_bytes(data=image_bytes, mime_type=mime_type))
                            else:
                                parts.append(Part.from_uri(image_url))
                else:
                    parts = [Part.from_text(message["content"])]
            else:
                parts = create_text_only_parts(message, include_placeholders=True)

            role = "user" if message["role"] == "user" else "model"
            content = Content(role=role, parts=parts)
            contents.append(content)

        if self.valves.PIPE_DEBUG:
            debug_print_contents(contents)

        return contents


    def _build_completion_payload(
        self,
        body: dict,
        __user__: dict,
        metadata: dict,
        user_valves: UserValves,
        emitter: EventEmitter,
    ) -> dict:
        logger.debug(f"Model from open-webui request: {body['model']}")
        manfold_name, model_name = self._parse_model_string(body['model'])

        logger.debug(f"User valves:\n{user_valves}")

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

        if body.get("title", False):  # If chat title generation is requested
            contents = [Content(role="user", parts=[Part.from_text(last_user_message_content)])]
        else:
            contents = self._build_conversation_history(messages)

        if self.valves.USE_PERMISSIVE_SAFETY or user_valves.USE_PERMISSIVE_SAFETY:
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            }
        else:
            safety_settings = body.get("safety_settings")

        tools=[]
        if self.valves.ENABLE_GROUNDING or user_valves.ENABLE_GROUNDING:
            logger.debug(f"Grounding enabled.")
            if model_name.startswith("gemini-2.0"):
                google_search_tool = Tool(
                    google_search = GoogleSearch()
                )
                tools=[google_search_tool]
            elif not model_name.startswith("gemini-exp"):
                google_search_tool = Tool(
                    google_search_retrieval = GoogleSearchRetrieval()
                )
                tools=[google_search_tool]

        generation_config = {
            "temperature": body.get("temperature", 0.7),
            "top_p": body.get("top_p", 0.9),
            "top_k": body.get("top_k", 40),
            "max_output_tokens": body.get("max_tokens", 8192),
            "stop_sequences": body.get("stop", []),
            "tools": tools,
            "response_modalities": ["TEXT"],
            "system_instruction": system_prompt,
            "safety_settings": safety_settings
        }
        config = GenerateContentConfig(
            **generation_config
        )
        payload = {
            "model": model_name,
            "contents": contents,
            "config": config
        }

        # Debug print full payload with readable conversation history
        debug_contents = []
        if system_message:
            debug_contents.append({"role": "system", "content": system_message["content"]})
        for content in contents:
            message = {"role": content.role}
            text_parts = []
            has_image = False
            for part in content.parts:
                if part.text:
                    text_parts.append(part.text)
                elif part.inline_data or part.file_data:
                    has_image = True
            message["content"] = " ".join(text_parts) if text_parts else ""
            if has_image:
                message["content"] = [message["content"], {"type": "image"}]
            debug_contents.append(message)
        
        # Convert config to dictionary for debug output
        debug_config = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_output_tokens": config.max_output_tokens,
            "stop_sequences": config.stop_sequences,
            "tools": config.tools,
            "response_modalities": config.response_modalities,
            "system_instruction": config.system_instruction,
            "safety_settings": config.safety_settings
        }

        debug_payload = {
            "model": payload["model"],
            "contents": debug_contents,
            "config": debug_config
        }
        logger.debug(f"Final payload:\n{pformat(debug_payload)}")

        return payload

    def _process_citations(self, response, citations):
        if hasattr(response, 'candidates') and response.candidates:
            logger.debug(f"Found response candidates")
            if hasattr(response.candidates[0], 'grounding_metadata') and response.candidates[0].grounding_metadata:
                logger.debug(f"Found grounding metadata")
                metadata = response.candidates[0].grounding_metadata
                if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks is not None:
                    logger.debug(f"Processing grounding chunks")
                    for grounding_chunk in metadata.grounding_chunks:
                        if hasattr(grounding_chunk, "web"):
                            citation = {
                                "title": grounding_chunk.web.title,
                                "uri": grounding_chunk.web.uri,
                            }
                            citations.append(citation)

    async def stream_response(self, response, citations):
        try:
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                self._process_citations(chunk, citations)

        except Exception as e:
            logger.error(f"Error in stream_response: {str(e)}")
            yield f"Error processing response: {str(e)}"


    async def get_response(self, response, citations):
        try:
            self._process_citations(response, citations)
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            return f"Error processing response: {str(e)}"

    def pipes(self):
        global logger

        if self.valves.PIPE_DEBUG:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging is enabled for the pipe")
        else:
            logger.setLevel(logging.INFO)
            logger.info("Debug logging is disabled for the pipe")

        self.client = google.genai.Client(
            api_key = self.valves.GOOGLE_API_KEY
        )
        return [
            {
                "id": "gemini-2.0-flash-exp",
                "name": f"{self.valves.NAME_PREFIX}gemini-2.0-flash-exp"
            },
            {
                "id": "gemini-exp-1206",
                "name": f"{self.valves.NAME_PREFIX}gemini-exp-1206"
            },
            {
                "id": "gemini-1.5-pro-latest",
                "name": f"{self.valves.NAME_PREFIX}gemini-1.5-pro-latest"
            },
            {
                "id": "gemini-1.5-pro-001",
                "name": f"{self.valves.NAME_PREFIX}gemini-1.5-pro-001"
            },
            {
                "id": "gemini-1.5-pro-002",
                "name": f"{self.valves.NAME_PREFIX}gemini-1.5-pro-002"
            },
        ]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __metadata__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
    ) -> AsyncGenerator[str, None]:
        try:
            logger.debug(f"pipe:{__name__}")

            user_valves = __user__.get("valves", self.UserValves())
            emitter = EventEmitter(__event_emitter__)

            payload = self._build_completion_payload(body, __user__, __metadata__, user_valves, emitter)

            if not self.valves.GOOGLE_API_KEY:
                raise Exception("GOOGLE_API_KEY not provided in valves.")

            citations = []

            if body["stream"]:
                response = self.client.models.generate_content_stream(
                    **payload
                )

                async for text in self.stream_response(response, citations):
                    yield text
            else:
                response = self.client.models.generate_content(
                    **payload
                )
                text = await self.get_response(response, citations)
                yield text

            if citations and len(citations) > 0:
                logger.debug(f"Appending citations: {pformat(citations)}")
                content = f"\n\n<details>\n<summary>Sources</summary>"
                for i, citation in enumerate(citations, 1):
                    content += f"\n[{i}] [{citation.get('title')}]({citation.get('uri')})"
                content += f"\n</details>\n"
                yield content

        except Exception as e:
            logger.debug(f"Error generating content: {e}")
            yield f"An error occurred: {str(e)}"