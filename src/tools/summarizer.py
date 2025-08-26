from llama_index.core.tools import FunctionTool
from llama_index.core.llms import LLM

class SummarizationTool:
    def __init__(self, llm: LLM):
        self.llm = llm

    async def summarize_text(self, text: str, summary_type: str = "concise") -> str:
        """Generate summary of provided text"""
        prompts = {
            "concise": "Provide a concise 2-3 sentence summary of the following text:",
            "detailed": "Provide a detailed summary covering main points and key details:",
            "bullet_points": "Summarize the following text as bullet points highlighting key information:"
        }
        prompt = f"{prompts.get(summary_type, prompts['concise'])}\n\nText: {text}"
        response = await self.llm.acomplete(prompt)
        return str(response)

def create_summarization_tool(llm: LLM):
    tool = SummarizationTool(llm)
    async def summarize(text: str, summary_type: str = "concise") -> str:
        """Generate summary of text with specified type"""
        return await tool.summarize_text(text, summary_type)
    return FunctionTool.from_defaults(
        async_fn=summarize,
        name="text_summarizer",
        description="Generate summaries of long documents or text passages"
    )