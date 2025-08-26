from llama_index.core.query_engine import RetrieverQueryEngine
from .retrievers import create_retriever

def create_query_engine(vector_store):
    retriever = create_retriever(vector_store)
    return RetrieverQueryEngine.from_args(retriever)
