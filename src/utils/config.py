import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Change to a faster model
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
# Use a smaller, faster embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
