from llama_index.core.memory import Memory, ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from typing import List, Optional

class ShortTermMemory:
    def __init__(self, session_id: str, token_limit: int = 4000):
        self.memory = Memory.from_defaults(
            session_id=session_id,
            token_limit=token_limit
        )

    async def add_message(self, role: str, content: str):
        """Add a message to short-term memory"""
        message = ChatMessage(role=role, content=content)
        self.memory.put_messages([message])

    async def get_context(self) -> List[ChatMessage]:
        """Retrieve conversation context"""
        return self.memory.get()

    async def clear_context(self):
        """Clear short-term memory"""
        self.memory.reset()