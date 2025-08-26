from llama_index.core.readers import SimpleDirectoryReader
from pathlib import Path

def load_documents():
    data_dir = Path(__file__).parent.parent.parent / "data" / "documents"
    reader = SimpleDirectoryReader(input_dir=data_dir)
    return reader.load_data()
