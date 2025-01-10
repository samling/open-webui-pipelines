"""
title: Google GenAI (Vertex AI) Manifold Pipeline
author: Sam B
date: 2024-12-18
version: 0.1.0
license: MIT
description: A pipeline for generating text using Google's GenAI models in Open-WebUI.
requirements: google, google-genai, google-cloud-aiplatform>=1.74.0
"""

import base64
import google
import google.genai
import logging
import os
import json

from google.genai.types import Tool, GenerateContentConfig, GoogleSearch, GoogleSearchRetrieval
from google.genai.types import (
    Content,
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
from pprint import pformat
from pydantic import BaseModel, Field
from typing import (
    Any,
    AsyncGenerator,
    Generator,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Union,
    )

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

class Pipeline:
    class Valves(BaseModel):
        """Options to change from the WebUI"""
        NAME_PREFIX: str 
        GOOGLE_API_KEY: str
        USE_PERMISSIVE_SAFETY: bool
        ENABLE_GROUNDING: bool
        PIPE_DEBUG: bool

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
        logger.debug("Initializing Pipeline")
        self.type = "manifold"
        
        # Load environment variables with debug logging
        env_vars = {
            "NAME_PREFIX": os.getenv('NAME_PREFIX', "VertexAI."),
            "GOOGLE_API_KEY": os.getenv('GOOGLE_API_KEY', "fake-key"),
            "USE_PERMISSIVE_SAFETY": os.getenv('USE_PERMISSIVE_SAFETY', False),
            "ENABLE_GROUNDING": os.getenv('ENABLE_GROUNDING', False),
            "PIPE_DEBUG": os.getenv('PIPE_DEBUG', False),
        }
        logger.debug(f"Loaded environment variables: {json.dumps({k: '***' if k == 'GOOGLE_API_KEY' else v for k,v in env_vars.items()})}")
        
        self.valves = self.Valves(**env_vars)
        logger.debug(f"Initialized valves: {self.valves.json(exclude={'GOOGLE_API_KEY'})}")
        
        self.user_valves = self.UserValves(
            **{
                "USE_PERMISSIVE_SAFETY": os.getenv('USE_PERMISSIVE_SAFETY', False),
                "ENABLE_GROUNDING": os.getenv('ENABLE_GROUNDING', False),
            }
        )
        logger.debug(f"Initialized user valves: {self.user_valves.json()}")
        
        self.client: google.genai.Client | None = None

        self.pipelines = self.get_models()

    def get_models(self):
        models = [
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
        logger.debug(f"Initialized pipelines: {json.dumps(models, indent=2)}")
        return models
        
    async def on_startup(self) -> None:
        logger.info("Starting pipeline initialization")
        
        if self.valves.PIPE_DEBUG:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging is enabled for the pipe")
        else:
            logger.setLevel(logging.INFO)
            logger.info("Debug logging is disabled for the pipe")

        try:
            logger.debug(f"Initializing Google GenAI client with API key: {'*' * len(self.valves.GOOGLE_API_KEY)}")
            self.client = google.genai.Client(
                api_key = self.valves.GOOGLE_API_KEY
            )
            logger.info("Successfully initialized Google GenAI client")
            pass
        except Exception as e:
            logger.error(f"Failed to initialize Google GenAI client: {str(e)}")
            raise

    async def on_valves_updated(self) -> None:
        self.pipelines = self.get_models()
        
        if self.valves.PIPE_DEBUG:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging is enabled for the pipe")
        else:
            logger.setLevel(logging.INFO)
            logger.info("Debug logging is disabled for the pipe")

        try:
            logger.debug(f"Reinitializing Google GenAI client with API key: {'*' * len(self.valves.GOOGLE_API_KEY)}")
            self.client = google.genai.Client(
                api_key = self.valves.GOOGLE_API_KEY
            )
            logger.info("Successfully reinitialized Google GenAI client")
        except Exception as e:
            logger.error(f"Failed to reinitialize Google GenAI client: {str(e)}")
            raise

        logger.info("Updating pipeline configuration")
        pass

    async def on_shutdown(self) -> None:
        logger.info("Shutting down pipeline")

    def _parse_model_string(self, model_id):
        """Parse model ID into manifold and model name components"""
        logger.debug(f"Parsing model string: {model_id}")
        
        parts = model_id.split(".", 1)
        manifold_name = None
        model_name = None

        if len(parts) == 1:
            model_name = parts[0]
        else:
            manifold_name = parts[0]
            model_name = parts[1]

        logger.debug(f"Parsed manifold name: {manifold_name}")
        logger.debug(f"Parsed model name: {model_name}")

        return manifold_name, model_name

    def _build_conversation_history(self, messages: List[dict]) -> List[Content]:
        logger.debug(f"Building conversation history from {len(messages)} messages")
        contents = []
        
        def debug_print_contents(contents: List[Content]):
            """Helper function to print contents in a familiar LLM format"""
            debug_list = []
            for content in contents:
                message = {"role": content.role}
                text_parts = [p.text for p in content.parts if p.text]
                if text_parts:
                    message["text"] = " ".join(text_parts)
                binary_parts = [p for p in content.parts if p.inline_data or p.file_data]
                if binary_parts:
                    message["has_image"] = True
                debug_list.append(message)
            
            logger.debug(f"Conversation history:\n{pformat(debug_list)}")

        def create_text_only_parts(message, include_placeholders=True) -> List[Part]:
            """Extract only text content from a message"""
            logger.debug(f"Creating text-only parts from message: {pformat(message)}")
            
            if isinstance(message.get("content"), list):
                text_parts = []
                for content in message["content"]:
                    logger.debug(f"Processing content part: {pformat(content)}")
                    if content["type"] == "text":
                        logger.debug(f"Adding text part: {content['text']}")
                        text_parts.append(Part.from_text(content["text"]))
                    elif content["type"] == "image_url" and include_placeholders:
                        logger.debug("Adding placeholder for image")
                        text_parts.append(Part.from_text("[binary_data]"))
                logger.debug(f"Created {len(text_parts)} text parts")
                return text_parts if text_parts else [Part.from_text("")]
            else:
                logger.debug(f"Creating single text part from content: {message['content']}")
                return [Part.from_text(message["content"])]

        for message in messages:
            logger.debug(f"Processing message: {pformat(message)}")
            
            if message["role"] == "system":
                logger.debug("Skipping system message")
                continue

            is_user = message["role"] == "user"
            is_current_message = message == messages[-1]
            logger.debug(f"Message is user: {is_user}, is current: {is_current_message}")

            if is_user and is_current_message:
                parts = []
                if isinstance(message.get("content"), list):
                    for content in message["content"]:
                        logger.debug(f"Processing content in current user message: {pformat(content)}")
                        if content["type"] == "text":
                            logger.debug(f"Adding text part: {content['text']}")
                            parts.append(Part.from_text(content["text"]))
                        elif content["type"] == "image_url" and is_user:
                            image_url = content["image_url"]["url"]
                            logger.debug(f"Processing image URL: {image_url[:50]}...")
                            if image_url.startswith("data:image"):
                                metadata, base64_data = image_url.split(",", 1)
                                mime_type = metadata.split(";")[0].split(":")[1]
                                image_bytes = base64.b64decode(base64_data)
                                logger.debug(f"Adding image with mime_type: {mime_type}")
                                parts.append(Part.from_bytes(data=image_bytes, mime_type=mime_type))
                            else:
                                logger.debug("Adding image from URI")
                                parts.append(Part.from_uri(image_url))
                else:
                    logger.debug(f"Adding single text part: {message['content']}")
                    parts = [Part.from_text(message["content"])]
            else:
                parts = create_text_only_parts(message, include_placeholders=True)

            role = "user" if message["role"] == "user" else "model"
            logger.debug(f"Creating Content with role {role} and {len(parts)} parts")
            content = Content(role=role, parts=parts)
            contents.append(content)

        if self.valves.PIPE_DEBUG:
            debug_print_contents(contents)

        logger.debug(f"Built conversation history with {len(contents)} contents")
        return contents

    def _build_completion_payload(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> dict:
        logger.debug(f"Building completion payload for model: {model_id}")
        logger.debug(f"User message: {user_message}")
        logger.debug(f"Request body: {pformat(body)}")
        
        # Parse model ID
        model_name = model_id
        logger.debug(f"Using model name: {model_name}")

        # Process system prompt
        system_message = next(
            (msg["content"] for msg in messages if msg["role"] == "system"), None
        )
        # logger.debug(f"System message: {system_message}")

        # Build contents
        if body.get("title", False):
            logger.debug("Building contents for title generation")
            contents = [Content(role="user", parts=[Part.from_text(user_message)])]
        else:
            logger.debug("Building conversation history contents")
            contents = self._build_conversation_history(messages)

        # Configure safety settings
        if self.valves.USE_PERMISSIVE_SAFETY:
            logger.debug("Using permissive safety settings")
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            }
        else:
            logger.debug("Using default safety settings")
            safety_settings = body.get("safety_settings")

        # Configure tools
        tools = []
        if self.valves.ENABLE_GROUNDING:
            logger.debug("Grounding is enabled, checking message content")

            # Check if the last message had an image in it
            last_message = messages[-1]
            logger.debug(f"Last message: {last_message}")
            if isinstance(last_message.get("content"), list):
                has_images = any(content["type"] == "image_url" for content in last_message["content"])
                if has_images:
                    logger.error("Grounding cannot be used with image inputs")
                    raise ValueError("Grounding cannot be used with image inputs. Please disable grounding or remove images from your message.")

            logger.debug("Configuring search tools based on model")
            if model_name.startswith("gemini-2.0"):
                logger.debug("Using GoogleSearch tool for gemini-2.0")
                google_search_tool = Tool(
                    google_search = GoogleSearch()
                )
                tools=[google_search_tool]
            elif not model_name.startswith("gemini-exp"):
                logger.debug("Using GoogleSearchRetrieval tool for non-experimental model")
                google_search_tool = Tool(
                    google_search_retrieval = GoogleSearchRetrieval()
                )
                tools=[google_search_tool]

        # Build generation config
        generation_config = {
            "temperature": body.get("temperature", 0.7),
            "top_p": body.get("top_p", 0.9),
            "top_k": body.get("top_k", 40),
            "max_output_tokens": body.get("max_tokens", 8192),
            "stop_sequences": body.get("stop", []),
            "tools": tools,
            "response_modalities": ["TEXT"],
            "system_instruction": system_message,
            "safety_settings": safety_settings
        }
        logger.debug(f"Generation config: {pformat(generation_config)}")
        
        config = GenerateContentConfig(**generation_config)
        
        # Assemble final payload
        payload = {
            "model": model_name,
            "contents": contents,
            "config": config
        }

        # Debug print full payload with readable conversation history
        debug_contents = []
        # if system_message:
        #     debug_contents.append({"role": "system", "content": system_message})
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
        logger.debug("Processing citations from response")
        try:
            if hasattr(response, 'candidates') and response.candidates:
                logger.debug("Found response candidates")
                if hasattr(response.candidates[0], 'grounding_metadata') and response.candidates[0].grounding_metadata:
                    logger.debug("Found grounding metadata")
                    metadata = response.candidates[0].grounding_metadata
                    if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks is not None:
                        logger.debug("Processing grounding chunks")
                        for chunk in metadata.grounding_chunks:
                            if hasattr(chunk, "web"):
                                citation = {
                                    "title": chunk.web.title,
                                    "uri": chunk.web.uri,
                                }
                                logger.debug(f"Adding citation: {pformat(citation)}")
                                citations.append(citation)
        except Exception as e:
            logger.error(f"Error processing citations: {str(e)}")

    def stream_response(self, response, citations):
        logger.debug("Starting to stream response")
        try:
            for chunk in response:
                if chunk.text:
                    logger.debug(f"Yielding chunk: {chunk.text[:50]}...")
                    yield chunk.text
                self._process_citations(chunk, citations)

        except Exception as e:
            logger.error(f"Error in stream_response: {str(e)}")
            error_message = f"Error processing response: {str(e)}"
            logger.error(error_message)
            yield error_message

    def get_response(self, response, citations):
        logger.debug("Getting complete response")
        try:
            self._process_citations(response, citations)
            text = response.text if hasattr(response, 'text') else str(response)
            logger.debug(f"Response text: {text[:100]}...")
            return text
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            error_message = f"Error processing response: {str(e)}"
            logger.error(error_message)
            return error_message

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Iterator]:
        logger.info(f"Starting pipe execution for model {model_id}")
        logger.debug(f"User message: {user_message}")
        logger.debug(f"Messages count: {len(messages)}")
        logger.debug(f"Request body: {pformat(body)}")
        
        try:
            payload = self._build_completion_payload(user_message, model_id, messages, body)

            if not self.valves.GOOGLE_API_KEY:
                logger.error("GOOGLE_API_KEY not provided in valves")
                raise Exception("GOOGLE_API_KEY not provided in valves.")

            citations = []

            if body.get("stream", True):
                logger.debug("Using streaming response")
                response = self.client.models.generate_content_stream(
                    **payload
                )

                logger.debug("Starting stream response generator")
                for text in self.stream_response(response, citations):
                    yield text
            else:
                logger.debug("Using non-streaming response")
                response = self.client.models.generate_content(
                    **payload
                )
                text = self.get_response(response, citations)
                yield text

            if citations and len(citations) > 0:
                logger.debug(f"Appending {len(citations)} citations")
                content = f"\n\n**Sources**"
                for i, citation in enumerate(citations, 1):
                    content += f"\n[{i}] [{citation.get('title')}]({citation.get('uri')})"
                yield content
                yield ""

        except Exception as e:
            logger.error(f"Error generating content: {str(e)}", exc_info=True)
            yield f"An error occurred: {str(e)}"