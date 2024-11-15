from typing import List, Optional
from pydantic import BaseModel
from schemas import OpenAIChatMessage
import requests
import langchain

"""
title: Basic Pipeline Test
author: sboynton
date: 2024-11-13
version: 1.1
license: MIT
description: Example of a filter pipeline that can be used to edit the form data before it is sent to LLM API.
requirements: requests,langchain
"""

class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
        pipelines: List[str] = []

        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 0

        # Add your custom parameters/configuration here e.g. API_KEY that you want user to configure etc.
        pass

    def __init__(self):
        self.type = "filter"
        self.name = "Filter"
        self.valves = self.Valves(**{"pipelines": ["*"]})

        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # This filter is applied to the form data BEFORE it is sent to the LLM API.
        print(f"inlet:{__name__}")

        return body
        
    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
    	# This filter is applied to the form data AFTER it is sent to the LLM API.
        print(f"outlet:{__name__}")
