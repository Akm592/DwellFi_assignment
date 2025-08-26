from llama_index.core.memory import (
    StaticMemoryBlock,
    FactExtractionMemoryBlock,
    VectorMemoryBlock
)
from typing import List
from llama_index.core.llms import ChatMessage, LLM
from .memory_blocks import ResearchContextMemoryBlock

class LongTermMemory:
    def __init__(self, vector_store, llm: LLM, embed_model):
        self.memory_blocks = [
            StaticMemoryBlock(
                name="system_info",
                static_content="I am a research assistant specialized in document analysis",
                priority=0
            ),
            FactExtractionMemoryBlock(
                name="facts",
                llm=llm,
                max_facts=100,
                priority=1
            ),
            VectorMemoryBlock(
                name="conversation_history",
                vector_store=vector_store,
                embed_model=embed_model,
                priority=2
            ),
            ResearchContextMemoryBlock(
                name="research_context",
                llm=llm # Pass llm to ResearchContextMemoryBlock
            )
        ]

    async def process_memory_flush(self, messages: List[ChatMessage]):
        """Process messages when short-term memory flushes"""
        for block in self.memory_blocks:
            await block._aput(messages)

    async def get_relevant_context(self, query: str) -> str:
        # This is a placeholder. In a real implementation, this would
        # query the memory blocks to get relevant context.
        # For now, we will just return the context from the ResearchContextMemoryBlock
        for block in self.memory_blocks:
            if block.name == "research_context":
                return await block._aget()
        return "Relevant long term context"