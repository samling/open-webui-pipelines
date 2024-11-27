"""
title: Langfuse Filter Pipeline
author: open-webui
date: 2024-05-30
version: 1.1
license: MIT
description: A filter pipeline that uses Langfuse.
requirements: langfuse
"""

from functools import cache
from langfuse.api.resources.commons.errors.unauthorized_error import UnauthorizedError
from pprint import pformat
from pydantic import BaseModel, Field
from typing import List, Optional
from utils.pipelines.main import get_last_assistant_message, get_last_user_message
import json
import langfuse
import os

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

async def log(message: str):
    print(f"Inlet: {message}")

class Pipeline:
    class Valves(BaseModel):
        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 0

        pipelines: List[str] = []

        # Valves
        secret_key: str
        public_key: str
        host: str
        extra_metadata: str
        extra_tags: str

    def __init__(self):
        # Initialize
        self.type = "filter"
        self.langfuse = None
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
                "secret_key": os.getenv("LANGFUSE_SECRET_KEY", "your-secret-key-here"),
                "public_key": os.getenv("LANGFUSE_PUBLIC_KEY", "your-public-key-here"),
                "host": os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com"),
                "extra_tags": os.getenv("EXTRA_TAGS", '["open-webui"]'),
                "extra_metadata": os.getenv("EXTRA_METADATA", '{"source": "open-webui"}')
            }
        )

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        self.set_langfuse()

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        self.set_langfuse()

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        self.langfuse.flush()
        pass

    def set_langfuse(self):
        try:
            self.langfuse = langfuse.Langfuse(
                secret_key=self.valves.secret_key,
                public_key=self.valves.public_key,
                host=self.valves.host,
                debug=False
            )
            self.langfuse.auth_check()
            print("Connected to Langfuse")
        except UnauthorizedError:
            print("Langfuse credentials incorrect.")
        except Exception as e:
            print(f"Langfuse error: {str(e)}")

    async def inlet(
        self,
        body: dict,
        __user__: dict,
        __metadata__: dict,
    ) -> dict:

        # if "chat_id" not in body:
        #     body["chat_id"] = uuid.uuid4()

        # metadata
        metadata = load_json_dict(self.valves.extra_metadata)
        __metadata__["tags"] = ["open-webui"]
        print(f"Included metadata: {__metadata__}")
        # if metadata:
        #     if __metadata__:
        #         for k, v in metadata.items():
        #             if k in __metadata__:
        #                 if isinstance(v, list) and isinstance(__metadata__[k], list):
        #                     __metadata__[k].extend(v)
        #                 elif isinstance(body["metadata"][k], list):
        #                     __metadata__[k].append(v)
        #                 elif isinstance(v, list):
        #                     __metadata__[k] = [__metadata__[k]] + v
        #             else:
        #                 __metadata__[k] = v
        #                 # await log(f"Extra_metadata of key '{k}' was already present in request. Value before: '{body['metadata'][k]}', value after: '{v}'")
        #         await log("Updated metadata")
        #     else:
        #         __metadata__ = metadata
        #         await log("Set metadata")
        # else:
        #     await log("No metadata specified")

        tags = load_json_list(self.valves.extra_tags)
        if tags:
            if "tags" in body:
                body["tags"] += tags
                await log("Updated tags")
            else:
                body["tags"] = tags
                await log("Set tags")
        else:
            await log("No tags specified")

        # also add as langfuse metadata
        # __metadata__["trace_metadata"] = body["metadata"].copy()

        # await log(pformat(body))
        print(f"inlet:{__name__}")
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")

        body["metadata"] = {"tags": ["open-webui"]}

        # await log(f"Output body: {pformat(body)}")
        return body