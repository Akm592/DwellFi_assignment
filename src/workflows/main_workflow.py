from llama_index.core.workflow import (
    Event, StartEvent, StopEvent, Workflow, step, Context
)
from typing import Dict, Any, List
from datetime import datetime
from llama_index.core.llms import ChatMessage # Add this import

class ResearchQueryEvent(Event):
    query: str
    context: Dict[str, Any]

class MemoryRetrievalEvent(Event):
    relevant_context: str
    query: str

class ToolExecutionEvent(Event):
    tool_results: Dict[str, Any]
    query: str

class MainResearchWorkflow(Workflow):
    """Main workflow orchestrating the research assistant"""
    def __init__(self, llm, tools, memory_system, query_engines, query_planning_workflow):
        super().__init__()
        self.llm = llm
        self.tools = tools
        self.memory_system = memory_system
        self.query_engines = query_engines
        self.query_planning_workflow = query_planning_workflow

    async def _assess_query_complexity(self, query: str) -> float:
        import re
        # More advanced heuristic
        question_words = ["what", "who", "when", "where", "why", "how", "summarize", "compare"]
        # Check for multiple question words or conjunctions that often link distinct ideas
        complex_indicators = re.findall(r'\b(and|or|but|compare|contrast)\b', query.lower())
        question_indicators = [word for word in question_words if word in query.lower()]

        # A more robust scoring system
        score = 0
        if len(query.split()) > 20:
            score += 0.5
        elif len(query.split()) > 10:
            score += 0.2

        score += len(complex_indicators) * 0.2
        score += len(question_indicators) * 0.1

        # Normalize score to be between 0 and 1
        normalized_score = min(score / 1.5, 1.0)

        return normalized_score

    async def _execute_direct_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder for direct query execution
        # In a real implementation, this would use a query engine to answer the query.
        engine = self.query_engines.get("default")
        if not engine:
            return {"result": "No default query engine available", "sources": []}
        
        short_term_context = await self.memory_system["short_term"].get_context()
        augmented_query = f"{short_term_context}\n\nQuery: {query}"
        result = await engine.aquery(augmented_query)
        return {
            "result": str(result),
            "sources": getattr(result, 'source_nodes', [])
        }

    def _format_tool_results(self, tool_results: Dict[str, Any]) -> str:
        formatted_results = []
        for key, value in tool_results.items():
            formatted_results.append(f"{key.replace('_', ' ').title()}: {value}")
        return "\n".join(formatted_results)

    @step
    async def initialize_session(self, ctx: Context, ev: StartEvent) -> ResearchQueryEvent:
        """Initialize session and prepare context"""
        query = ev.query
        user_id = getattr(ev, 'user_id', 'default_user')

        # Retrieve relevant memory context
        short_term_context_messages = await self.memory_system["short_term"].get_context()
        short_term_context = "\n".join([f"{m.role}: {m.content}" for m in short_term_context_messages])
        long_term_context = await self.memory_system["long_term"].get_relevant_context(query)
        context = {
            "user_id": user_id,
            "short_term": short_term_context,
            "long_term": long_term_context,
            "timestamp": datetime.now()
        }
        # FIX: Address deprecation warning for ctx.set
        await ctx.store.set("session_context", context)
        return ResearchQueryEvent(query=query, context=context)

    @step
    async def process_query(self, ctx: Context, ev: ResearchQueryEvent) -> ToolExecutionEvent:
        """Process query using appropriate tools and engines"""
        query = ev.query
        context = ev.context
        # Determine if query needs decomposition
        complexity_score = await self._assess_query_complexity(query)
        if complexity_score > 0.7:
            # Use query planning workflow for complex queries
            planning_result = await self.query_planning_workflow.run(query=query)
            tool_results = {"planning_result": str(planning_result)}
        else:
            # Direct processing for simple queries
            tool_results = await self._execute_direct_query(query, ev.context)
        return ToolExecutionEvent(tool_results=tool_results, query=query)

    @step
    async def generate_response(self, ctx: Context, ev: ToolExecutionEvent) -> StopEvent:
        """Generate final response and update memory"""
        query = ev.query
        tool_results = ev.tool_results
        # FIX: Address deprecation warning for ctx.get
        session_context = await ctx.store.get("session_context")
        # Generate response using LLM
        response_prompt = f"""
Query: {query}
Available context and tool results:
{self._format_tool_results(tool_results)}
Conversation context:
{session_context.get('short_term', '')}
Relevant background:
{session_context.get('long_term', '')}
Provide a comprehensive, helpful response that directly addresses the query.
"""
        response = await self.llm.acomplete(response_prompt)
        final_response = str(response)
        
        # Update short-term memory
        await self.memory_system["short_term"].add_message("user", query)
        await self.memory_system["short_term"].add_message("assistant", final_response)
        
        # --- FIX: UPDATE LONG-TERM MEMORY ---
        # This step was missing. It processes the latest conversation turn
        # and allows the long-term memory blocks (like ResearchContextMemoryBlock) to update.
        await self.memory_system["long_term"].process_memory_flush(
            [ChatMessage(role="user", content=query), ChatMessage(role="assistant", content=final_response)]
        )
        
        return StopEvent(result={
            "response": final_response,
            "sources": tool_results.get("sources", []),
            "query": query
        })