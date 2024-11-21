"""
title: Query Milvus database using haystack
author: samling with heavy inspiration from constLiakos, justinh-rahb and ther3zz
funding_url: https://github.com/open-webui
version: 0.1.0
license: MIT
requirements: haystack-ai,pymilvus==2.4.9,milvus-haystack
"""

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
            default="http://milvus:19530",
            description="The Milvus endpoint to query.",
        )
        MILVUS_OPENAI_BASE_URL: str = Field(
            default="https://api.openai.com",
            description="The OpenAI base endpoint.",
        )
        MILVUS_OPENAI_API_KEY: str = Field(
            default="sk-fake-key",
            description="The OpenAI API key.",
        )
        MILVUS_OPENAI_TEXT_EMBEDDER: str = Field(
            default="text-embedding-3-small",
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

    async def query_milvus(
        self,
        query: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Send a search query to Milvus and get the content of the result. Search documents, images, pdfs, CSVs, etc.
        :params query: Search query sent to Milvus.
        :return: The content of the response.
        """
        try:
            emitter = EventEmitter(__event_emitter__)

            await emitter.emit(f"Initiating RAG search for: {query}")

            prompt_template = """Answer the following query based on the provided context. If the context does
                not include an answer, reply with {{documents}}.\n
                Query: {{query}}
                Documents:
                {% for doc in documents %}
                {{ doc.content }}
                {% endfor %}
                Answer: 
            """

            milvus_endpoint = self.valves.MILVUS_ENDPOINT
            milvus_openai_text_embedder = self.valves.MILVUS_OPENAI_TEXT_EMBEDDER
            milvus_top_k = self.valves.MILVUS_TOP_K

            document_store = MilvusDocumentStore(
                connection_args={"uri": milvus_endpoint}
            )

            rag_pipeline = Pipeline()
            rag_pipeline.add_component("text_embedder", OpenAITextEmbedder(model=milvus_openai_text_embedder))
            rag_pipeline.add_component("retriever", MilvusEmbeddingRetriever(document_store=document_store, top_k=milvus_top_k))

            rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

            try:
                await emitter.emit("Sending request to Milvas")
                
                results = rag_pipeline.run(
                    {
                        "text_embedder": {"text": query},
                        "prompt_builder": {"query": query}
                    }
                )
                await emitter.emit(f"Retrieved query results")
                return json.dumps(results['retriever']['documents'])

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