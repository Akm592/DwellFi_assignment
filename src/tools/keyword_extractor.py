from llama_index.core.tools import FunctionTool
import yake
from keybert import KeyBERT
from typing import List, Tuple, Dict

class KeywordExtractionTool:
    def __init__(self):
        self.yake_extractor = yake.KeywordExtractor(
            lan="en", n=3, dedupLim=0.7, top=20
        )
        self.keybert_extractor = KeyBERT()

    def extract_keywords_yake(self, text: str, max_keywords: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords using YAKE algorithm"""
        keywords = self.yake_extractor.extract_keywords(text)
        return keywords[:max_keywords]

    def extract_keywords_bert(self, text: str, max_keywords: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords using KeyBERT"""
        return self.keybert_extractor.extract_keywords(
            text, keyphrase_ngram_range=(1, 2), stop_words='english', top_k=max_keywords
        )

    def extract_comprehensive_keywords(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        """Extract keywords using multiple methods"""
        return {
            "yake": self.extract_keywords_yake(text),
            "keybert": self.extract_keywords_bert(text)
        }

# Convert to LlamaIndex tool
def create_keyword_extraction_tool():
    extractor = KeywordExtractionTool()
    def extract_keywords(text: str, method: str = "comprehensive") -> str:
        """Extract keywords from text using specified method"""
        if method == "yake":
            keywords = extractor.extract_keywords_yake(text)
        elif method == "keybert":
            keywords = extractor.extract_keywords_bert(text)
        else:
            keywords = extractor.extract_comprehensive_keywords(text)
        return f"Extracted keywords: {keywords}"
    return FunctionTool.from_defaults(
        fn=extract_keywords,
        name="keyword_extractor",
        description="Extract important keywords and phrases from text documents"
    )