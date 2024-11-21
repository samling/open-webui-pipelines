"""
title: Query Milvus database using haystack
author: samling with heavy inspiration from constLiakos, justinh-rahb and ther3zz
funding_url: https://github.com/open-webui
version: 0.1.0
license: MIT
requirements: haystack-ai,pymilvus==2.4.9,milvus-haystack
"""

import asyncio
import os
import json
from pydantic import BaseModel, Field
from typing import Callable, Any
from haystack import Pipeline
from haystack.components.embedders import OpenAITextEmbedder
from milvus_haystack import MilvusDocumentStore
from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever

class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def emit(self, description="Unknown State", status="in_progress", done=False):
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


class Tools:
    class Valves(BaseModel):
        MILVUS_ENDPOINT: str = Field(
            default="http://milvus.milvus:19530",
            description="The Milvus endpoint to query.",
        )
        MILVUS_OPENAI_BASE_URL: str = Field(
            default="http://litellm.litellm:4000",
            description="The OpenAI base endpoint.",
        )
        MILVUS_OPENAI_API_KEY: str = Field(
            default="sk-fake-key",
            description="The OpenAI API key.",
        )
        MILVUS_OPENAI_TEXT_EMBEDDER: str = Field(
            default="text-embedding-3-large",
            description="The embedding model from OpenAI (or compatible endpoint) to use.",
        )
        MILVUS_TOP_K: int = Field(
            default=3,
            description="The number of results to consider.",
        )

    def __init__(self):
        self.valves = self.Valves()
        # self.user_valves = self.UserValves()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        os.environ['OPENAI_API_KEY'] = self.valves.MILVUS_OPENAI_API_KEY
        os.environ['OPENAI_BASE_URL'] = self.valves.MILVUS_OPENAI_BASE_URL

    async def query(
        self,
        query: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Search the database and send the contents of the result. Search for unknown knowledge, news, info, public contact info, weather, etc..
        :params query: Query sent to Milvus.
        :return: The content of the response.
        """
        try:
            emitter = EventEmitter(__event_emitter__)

            await emitter.emit(f"Initiating RAG search for: {query}")

            print(f"ENDPOINT: {self.valves.MILVUS_ENDPOINT}")

            milvus_endpoint = self.valves.MILVUS_ENDPOINT
            milvus_openai_base_url = self.valves.MILVUS_OPENAI_BASE_URL
            milvus_openai_api_key = self.valves.MILVUS_OPENAI_API_KEY
            milvus_openai_text_embedder = self.valves.MILVUS_OPENAI_TEXT_EMBEDDER
            milvus_top_k = self.valves.MILVUS_TOP_K

            print("WE ARE RIGHT BEFORE THE DOCUMENT STORE INIT")
            document_store = MilvusDocumentStore(
                connection_args={"uri": milvus_endpoint}
            )
            print("WE ARE RIGHT BEFORE THE PIPELINE INIT")
            rag_pipeline = Pipeline()
            print("WE ARE RIGHT BEFORE THE ADD COMPONENT")
            rag_pipeline.add_component(
                "text_embedder",
                OpenAITextEmbedder(
                    model=milvus_openai_text_embedder,
                    # api_base_url=milvus_openai_base_url,
                    # api_key=milvus_openai_api_key,
                ),
            )
            print("WE ARE RIGHT BEFORE THE ADD COMPONENT 2")
            rag_pipeline.add_component(
                "retriever",
                MilvusEmbeddingRetriever(
                    document_store=document_store, top_k=milvus_top_k
                ),
            )

            print("WE ARE RIGHT BEFORE THE PIPELINE CONNECT")
            rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

            print("WE ARE RIGHT BEFORE THE TRY BLOCK")
            try:
                await emitter.emit("Sending request to Milvas")
                print("WE ARE HERE")

                results = rag_pipeline.run(
                    {
                        "text_embedder": {"text": query},
                    }
                )
                await emitter.emit(f"Retrieved query results")
                return {"results": results["retriever"]["documents"]}

            except Exception as e:
                await emitter.emit(
                    status="error",
                    description=f"Error during search: {str(e)}",
                    done=True,
                )
                return json.dumps({"error": str(e)})

        except Exception as e:
            return {
                "content": f"Unexpected error occurred. Error: {str(e)}",
            }

async def main():
    tools = Tools()
    results = await tools.query("What are all the transactions in October 2024?")
 
    print(results)
 
 
if __name__ == "__main__":
    asyncio.run(main())
