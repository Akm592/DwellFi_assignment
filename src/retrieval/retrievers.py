from llama_index.core import VectorStoreIndex
from .document_loader import load_documents

def create_retriever(vector_store):
    if vector_store._collection.count() == 0:
        documents = load_documents()
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store,
        )
    else:
        index = VectorStoreIndex.from_vector_store(vector_store)
    return index.as_retriever()