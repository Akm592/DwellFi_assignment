from llama_index.core.memory import BaseMemoryBlock
from llama_index.core.llms import ChatMessage, LLM
from typing import List, Optional, Dict, Any
from src.tools.keyword_extractor import KeywordExtractionTool
from pydantic import Field

class ResearchContextMemoryBlock(BaseMemoryBlock[str]):
    """Custom memory block for research context tracking"""
    llm: Optional[LLM] = None
    research_topics: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    keyword_extractor: KeywordExtractionTool = Field(default_factory=KeywordExtractionTool)

    def __init__(self, name: str = "research_context", llm: Optional[LLM] = None):
        # pass both name and llm to pydantic's BaseModel init
        super().__init__(name=name, llm=llm)

    async def _extract_research_topics(self, content: str) -> Dict[str, Any]:
        # Use the keyword extraction tool to identify research topics.
        keywords = self.keyword_extractor.extract_keywords_yake(content)
        return {"topics": [kw[0] for kw in keywords]}

    async def _aget(self, messages: Optional[List[ChatMessage]] = None, **kwargs) -> str:
        context = []
        if self.research_topics:
            context.append(f"Current research topics: {', '.join(self.research_topics.get('topics', []))}")
        if self.user_preferences:
            context.append(f"User preferences: {self.user_preferences}")
        return "\n".join(context)

    async def _aput(self, messages: List[ChatMessage]) -> None:
        # Extract research topics and preferences from messages
        for message in messages:
            # --- FIX: Extract topics from the USER'S message ---
            # This is more reliable for capturing user intent.
            if message.role == "user" and "research" in message.content.lower():
                # Extract research topics using keyword extraction
                topics = await self._extract_research_topics(message.content)
                self.research_topics.update(topics)