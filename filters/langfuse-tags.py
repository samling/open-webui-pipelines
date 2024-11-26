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
from typing import List, Optional
import json
import langfuse
import os
import uuid

from pydantic import BaseModel

@cache
def load_json_dict(user_value: str) -> dict:
    user_value = user_value.strip()
    if not user_value:
        return {}
    loaded = json.loads(user_value)
    assert isinstance(loaded, dict), f"json is not a dict but '{type(loaded)}'"
    return loaded

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
        tags: List[str]

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
                "tags": ["open-webui"]
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
        user: Optional[dict] = None
    ) -> dict:

        # if "chat_id" not in body:
        #     body["chat_id"] = uuid.uuid4()

        body["tags"] = ["open-webui"]

        async def log(message: str):
            print(f"Inlet: {message}")

        await log(f"Body: {pformat(body)}")

        print(f"inlet:{__name__}")
        return body

    # async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
    #     print(f"outlet:{__name__}")
    #     return body