from llama_index.core.workflow import (
    Event, StartEvent, StopEvent, Workflow, step, Context
)
from typing import List, Dict, Any
import re

class QueryDecompositionEvent(Event):
    query: str
    sub_queries: List[str]

class SubQueryExecutionEvent(Event):
    sub_query: str
    results: Dict[str, Any]

class SubQueriesExecutedEvent(Event):
    sub_results: List[Dict[str, Any]]

class QuerySynthesisEvent(Event):
    sub_results: List[Dict[str, Any]]
    final_response: str

class QueryPlanningWorkflow(Workflow):
    """Intelligent query planning and decomposition workflow"""
    def __init__(self, llm, query_engines: Dict[str, Any]):
        super().__init__()
        self.llm = llm
        self.query_engines = query_engines

    def _extract_sub_queries(self, response: str) -> List[str]:
        # Extract numbered list items as sub-queries
        return [line.strip() for line in re.findall(r"^\d+\.\s*(.*)", response, re.MULTILINE)]

    async def _select_query_engine(self, sub_query: str) -> Any:
        # For now, just return the default query engine
        return self.query_engines.get("default")

    def _format_sub_results(self, sub_results: List[Dict[str, Any]]) -> str:
        formatted_results = []
        for res in sub_results:
            formatted_results.append(f"Sub-query: {res['query']}\nResult: {res['result']}")
            if res.get('sources'):
                formatted_results.append(f"Sources: {res['sources']}")
        return "\n\n".join(formatted_results)

    @step
    async def plan_query(self, ctx: Context, ev: StartEvent) -> QueryDecompositionEvent:
        """Decompose complex query into manageable sub-queries"""
        query = ev.query
        planning_prompt = f"""
Given this complex query: "{query}"
Break it down into 3-5 specific sub-questions that can be answered independently.
Each sub-question should focus on a specific aspect of the main query.
Format as a numbered list:
1. [Sub-question 1]
2. [Sub-question 2]
... 
"""
        response = await self.llm.acomplete(planning_prompt)
        sub_queries = self._extract_sub_queries(str(response))
        # FIX: Address deprecation warning for ctx.set
        await ctx.store.set("original_query", query)
        return QueryDecompositionEvent(query=query, sub_queries=sub_queries)

    @step
    async def execute_sub_queries(
        self, ctx: Context, ev: QueryDecompositionEvent
    ) -> SubQueriesExecutedEvent:
        """Execute each sub-query using appropriate tools/engines"""
        sub_results = []
        for sub_query in ev.sub_queries:
            # Determine best engine/tool for this sub-query
            engine = await self._select_query_engine(sub_query)
            if not engine:
                sub_results.append({
                    "query": sub_query,
                    "error": "No suitable query engine found",
                    "result": "Unable to process this sub-query"
                })
                continue
            try:
                result = await engine.aquery(sub_query)
                sub_results.append({
                    "query": sub_query,
                    "result": str(result),
                    "sources": getattr(result, 'source_nodes', [])
                })
            except Exception as e:
                sub_results.append({
                    "query": sub_query,
                    "error": str(e),
                    "result": "Unable to process this sub-query"
                })
        return SubQueriesExecutedEvent(sub_results=sub_results)

    @step
    async def synthesize_results(
        self, ctx: Context, ev: SubQueriesExecutedEvent
    ) -> StopEvent:
        """Synthesize sub-query results into final comprehensive answer"""
        # FIX: Address deprecation warning for ctx.get
        original_query = await ctx.store.get("original_query")
        synthesis_prompt = f"""
    You are a helpful research assistant. Your task is to synthesize the following results from a series of sub-queries into a single, comprehensive answer to the user's original question.

    **Original Question:** "{original_query}"

    **Collected Evidence (from sub-queries):**
    {self._format_sub_results(ev.sub_results)}

    **Instructions:**
    1.  Carefully review the collected evidence.
    2.  Formulate a clear, coherent, and well-structured response that directly answers the **Original Question**.
    3.  Integrate the information from the sub-query results naturally into your answer. Do not simply list the results.
    4.  If sources are available, cite them appropriately within the text.
    5.  Do not include information that is not supported by the provided evidence.

    **Final Answer:**
    """
        final_response = await self.llm.acomplete(synthesis_prompt)
        return StopEvent(result=str(final_response))