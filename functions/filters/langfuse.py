"""
title: Langfuse Filter Pipeline
author: open-webui
date: 2024-09-27
version: 1.4
license: MIT
description: A filter pipeline that uses Langfuse.
requirements: langfuse
"""

from typing import List, Optional
import os
import uuid

from open_webui.utils.misc import get_last_assistant_message
from pprint import pformat
from pydantic import BaseModel, Field
from langfuse import Langfuse
from langfuse.api.resources.commons.errors.unauthorized_error import UnauthorizedError


def get_last_assistant_message_obj(messages: List[dict]) -> dict:
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message
    return {}


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=0, description="The priority of this pipe.")
        secret_key: str = Field(default="", description="The Langfuse secret key.")
        public_key: str = Field(default="", description="The Langfuse public key.")
        host: str = Field(
            default="http://langfuse-web.langfuse:3000",
            description="The Langfuse host.",
        )
        extra_tags: str = Field(
            default="open-webui-test", description="Extra tags for the traces."
        )
        debug: bool = Field(default=False, description="Enable Langfuse debugging.")

    def __init__(self):
        self.name = "Langfuse Filter"
        self.valves = self.Valves()
        self.langfuse = None
        self.chat_generations = {}
        self.set_langfuse()

    # async def on_startup(self):
    #     print(f"on_startup:{__name__}")
    #     self.set_langfuse()

    # async def on_shutdown(self):
    #     print(f"on_shutdown:{__name__}")
    #     self.langfuse.flush()

    async def on_valves_updated(self):
        self.set_langfuse()

    def set_langfuse(self):
        try:
            self.langfuse = Langfuse(
                secret_key=self.valves.secret_key,
                public_key=self.valves.public_key,
                host=self.valves.host,
                tags=[self.valves.extra_tags],
                debug=bool(self.valves.debug),
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

    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        print(f"inlet:{__name__}")
        print(f"Received body: {pformat(body)}")
        print(f"User: {pformat(__user__)}")

        # Check for presence of required keys and generate chat_id if missing
        if "chat_id" not in body:
            unique_id = f"SYSTEM MESSAGE {uuid.uuid4()}"
            body["chat_id"] = unique_id
            print(f"chat_id was missing, set to: {unique_id}")

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
            session_id=body["chat_id"],
        )

        generation = trace.generation(
            name=body["chat_id"],
            model=body["model"],
            input=body["messages"],
            metadata={"interface": "open-webui"},
        )

        self.chat_generations[body["chat_id"]] = generation
        print(trace.get_trace_url())

        return body

    async def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")
        print(f"Received body: {body}")
        if body["chat_id"] not in self.chat_generations:
            return body

        generation = self.chat_generations[body["chat_id"]]
        assistant_message = get_last_assistant_message(body["messages"])

        # Extract usage information for models that support it
        usage = None
        assistant_message_obj = get_last_assistant_message_obj(body["messages"])
        if assistant_message_obj:
            info = assistant_message_obj.get("info", {})
            if isinstance(info, dict):
                input_tokens = info.get("prompt_eval_count") or info.get(
                    "prompt_tokens"
                )
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
        del self.chat_generations[body["chat_id"]]

        return body
