"""
title: Milvus RAG Hybrid Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Haystack library.
requirements: haystack-ai,pymilvus==2.4.9,milvus-haystack,fastembed-haystack
"""

from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
import os

class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str
        OPENAI_BASE_URL: str
        MILVUS_URL: str
        MILVUS_EMBEDDING_MODEL: str
        MILVUS_GENERATOR_MODEL: str
        MILVUS_TOP_K: int

    def __init__(self):
        self.rag_pipeline = None
        self.valves = self.Valves(
            **{
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "sk-fake-key"),
                "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", "http://openai"),
                "MILVUS_URL": os.getenv("MILVUS_URL", "http://milvus.milvus:19530"),
                "MILVUS_EMBEDDING_MODEL": os.getenv("MILVUS_EMBEDDING_MODEL", "text-embedding-3-large"),
                "MILVUS_GENERATOR_MODEL": os.getenv("MILVUS_QUERY_MODEL", "gpt-4o"),
                "MILVUS_TOP_K": os.getenv("MILVUS_TOP_K", 10),
            }
        )

    async def on_startup(self):
        os.environ["OPENAI_API_KEY"] = self.valves.OPENAI_API_KEY
        os.environ['OPENAI_BASE_URL'] = self.valves.OPENAI_BASE_URL

        from haystack import Pipeline
        from haystack.components.embedders import OpenAITextEmbedder
        from haystack.components.builders import PromptBuilder
        from haystack.components.generators import OpenAIGenerator
        from milvus_haystack import MilvusDocumentStore
        from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever, MilvusHybridRetriever
        from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder

        prompt_template = """
        Given the following information, answer the question.

        Query: {{question}}

        Documents:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Answer:
        """

        document_store = MilvusDocumentStore(
            connection_args={"uri": self.valves.MILVUS_URL},
            sparse_vector_field="sparse_vector",
        )
        self.rag_pipeline = Pipeline()
        self.rag_pipeline.add_component("sparse_text_embedder", FastembedSparseTextEmbedder())
        self.rag_pipeline.add_component("dense_text_embedder", OpenAITextEmbedder(model=self.valves.MILVUS_EMBEDDING_MODEL))
        self.rag_pipeline.add_component("retriever", MilvusHybridRetriever(document_store=document_store, top_k=self.valves.MILVUS_TOP_K))
        self.rag_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
        self.rag_pipeline.add_component("generator", OpenAIGenerator(generation_kwargs={"temperature": 0}, model=self.valves.MILVUS_GENERATOR_MODEL))

        self.rag_pipeline.connect("sparse_text_embedder.sparse_embedding", "retriever.query_sparse_embedding")
        self.rag_pipeline.connect("dense_text_embedder.embedding", "retriever.query_embedding")
        self.rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        self.rag_pipeline.connect("prompt_builder", "generator")

        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(messages)
        print(user_message)

        question = user_message
        response = self.rag_pipeline.run(
            {
                "dense_text_embedder": {"text": question},
                "sparse_text_embedder": {"text": question},
                "prompt_builder": {"question": question},
            }
        )

        return response["generator"]["replies"][0]
