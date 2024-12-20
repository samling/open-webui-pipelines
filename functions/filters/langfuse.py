"""
title: Langfuse Filter Pipeline
author: open-webui
date: 2024-09-27
version: 0.1.0
license: MIT
description: A filter pipeline that uses Langfuse.
requirements: langfuse
"""

from typing import List, Optional
import os
import logging
import uuid

from open_webui.utils.misc import get_last_assistant_message
from pprint import pformat
from pydantic import BaseModel, Field
from langfuse import Langfuse
from langfuse.api.resources.commons.errors.unauthorized_error import UnauthorizedError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def get_last_assistant_message_obj(messages: List[dict]) -> dict:
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message
    return {}


class Filter:
    class Valves(BaseModel):
        PRIORITY: int = Field(default=0, description="The priority of this pipe.")
        SECRET_KEY: str = Field(default="", description="The Langfuse secret key.")
        PUBLIC_KEY: str = Field(default="", description="The Langfuse public key.")
        HOST: str = Field(
            default="http://langfuse-web.langfuse:3000",
            description="The Langfuse host.",
        )
        EXTRA_TAGS: str = Field(
            default="open-webui-test", description="Extra tags for the traces."
        )
        LANGFUSE_DEBUG: bool = Field(default=False, description="Enable Langfuse debugging.")
        PIPE_DEBUG: bool = Field(default=False, description="Enable pipe debugging.")

    def __init__(self):
        self.name = "Langfuse Filter"
        self.valves = self.Valves()
        self.langfuse = None
        self.chat_generations = {}

    def _set_langfuse(self):
        try:
            self.langfuse = Langfuse(
                secret_key=self.valves.SECRET_KEY,
                public_key=self.valves.PUBLIC_KEY,
                host=self.valves.HOST,
                debug=bool(self.valves.LANGFUSE_DEBUG),
            )
            self.langfuse.auth_check()
        except UnauthorizedError:
            print(
                "Langfuse credentials incorrect. Please re-enter your Langfuse credentials in the pipeline settings."
            )
        except Exception as e:
            print(
                f"Langfuse error: {e} Please re-enter your Langfuse credentials in the pipeline settings."
            )

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
    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        print(f"inlet:{__name__}")
        print(f"Received inlet body: {pformat(body)}")
        print(f"User: {pformat(__user__)}")

        if self.valves.PIPE_DEBUG:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging is enabled for the inlet")
        else:
            logger.setLevel(logging.INFO)
            logger.info("Debug logging is disabled for the inlet")

        self._set_langfuse()

        manifold_name, model_name = self._parse_model_string(body['model'])

        # Check for presence of required keys and generate chat_id if missing
        session_id = None
        if "chat_id" not in body['metadata']:
            session_id = f"SYSTEM MESSAGE {uuid.uuid4()}"
            body['metadata']["chat_id"] = session_id
            print(f"chat_id was missing, set to: {session_id}")
        else:
            session_id = body['metadata']['chat_id']
            print(f"chat_id is {session_id}")

        required_keys = ["model", "messages"]
        missing_keys = [key for key in required_keys if key not in body]

        if missing_keys:
            error_message = (
                f"Error: Missing keys in the request body: {', '.join(missing_keys)}"
            )
            print(error_message)
            raise ValueError(error_message)

        trace = self.langfuse.trace(
            name=f"filter:{__name__}",
            input=body,
            user_id=__user__["email"],
            metadata={"user_name": __user__["name"], "user_id": __user__["id"]},
            session_id=session_id,
            tags=[self.valves.EXTRA_TAGS],
        )

        generation = trace.generation(
            name=body['metadata']["chat_id"],
            model=model_name,
            input=body["messages"],
            metadata={"interface": "open-webui"},
        )

        self.chat_generations[session_id] = generation
        print(f"Trace URL: {trace.get_trace_url()}")

        return body

    async def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")
        print(f"Received outlet body: {body}")

        if self.valves.PIPE_DEBUG:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging is enabled for the outlet")
        else:
            logger.setLevel(logging.INFO)
            logger.info("Debug logging is disabled for the outlet")

        session_id = None
        if body["chat_id"] not in self.chat_generations:
            return body
        else:
            session_id = body['chat_id']

        generation = self.chat_generations[session_id]
        assistant_message = get_last_assistant_message(body["messages"])

        # Extract usage information for models that support it
        usage = None
        assistant_message_obj = get_last_assistant_message_obj(body["messages"])
        if assistant_message_obj:
            info = assistant_message_obj.get("info", {})
            if isinstance(info, dict):
                input_tokens = info.get("prompt_eval_count") or info.get("prompt_tokens")
                output_tokens = info.get("eval_count") or info.get("completion_tokens")
                if input_tokens is not None and output_tokens is not None:
                    usage = {
                        "input": input_tokens,
                        "output": output_tokens,
                        "unit": "TOKENS",
                    }

        # Update generation
        generation.end(
            output=assistant_message,
            metadata={"interface": "open-webui"},
            usage=usage,
        )

        # Clean up the chat_generations dictionary
        del self.chat_generations[session_id]

        return body
