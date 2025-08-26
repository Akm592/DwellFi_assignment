import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import chromadb
from llama_index.core.llms import ChatMessage
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


from src.workflows.main_workflow import MainResearchWorkflow
from src.memory.short_term_memory import ShortTermMemory
from src.memory.long_term_memory import LongTermMemory
from src.tools.keyword_extractor import create_keyword_extraction_tool
from src.tools.summarizer import create_summarization_tool
from src.retrieval.query_engine import create_query_engine
from src.utils.config import GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL
from src.utils.logging_setup import setup_logging

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from llama_index.core import Settings
from src.workflows.query_planning_workflow import QueryPlanningWorkflow

import hashlib
from src.utils.caching import CacheManager


# Setup logging
setup_logging()

# Create a dictionary to hold our application's state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Ran on startup ---
    print("Initializing core components...")
    # Initialize components that can be shared across requests
    llm = Groq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

    Settings.llm = llm
    Settings.embed_model = embed_model

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("research-assistant-collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Store components in the app_state dictionary
    app_state["long_term_memory"] = LongTermMemory(vector_store=vector_store, llm=llm, embed_model=embed_model)
    app_state["query_engine"] = create_query_engine(vector_store)
    app_state["tools"] = [
        create_keyword_extraction_tool(),
        create_summarization_tool(llm=llm)
    ]
    # IMPORTANT: Initialize the QueryPlanningWorkflow
    app_state["query_planning_workflow"] = QueryPlanningWorkflow(llm=llm, query_engines={"default": app_state["query_engine"]})
    app_state["cache_manager"] = CacheManager()

    print("Initialization complete.")
    yield
    # --- Ran on shutdown ---
    app_state.clear()
    print("Application shutdown and cleanup complete.")


# Initialize FastAPI app with the lifespan manager
app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session"


@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        # Create a unique hash for the query to use as a cache key
        query_hash = hashlib.sha256(request.query.encode()).hexdigest()
        cache_manager = app_state["cache_manager"]

        # 1. Check for a cached response first
        cached_response = await cache_manager.get_cached_response(query_hash)
        if cached_response:
            print("Returning response from cache.")
            return cached_response
            
        # Create a new ShortTermMemory for each request to maintain session state
        short_term_memory = ShortTermMemory(session_id=request.session_id)

        # Create a new MainResearchWorkflow, but reuse the heavy components
        main_workflow = MainResearchWorkflow(
            llm=Settings.llm,
            tools=app_state["tools"],
            memory_system={"short_term": short_term_memory, "long_term": app_state["long_term_memory"]},
            query_engines={"default": app_state["query_engine"]},
            # Pass the initialized planning workflow
            query_planning_workflow=app_state["query_planning_workflow"]
        )

        result = await main_workflow.run(query=request.query, user_id=request.session_id)
        
        # 2. Cache the new response before returning
        await cache_manager.cache_response(query_hash, result)
        print("Response cached.")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))