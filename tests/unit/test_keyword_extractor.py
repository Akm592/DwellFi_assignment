import pytest
from src.tools.keyword_extractor import KeywordExtractionTool
from unittest.mock import MagicMock

@pytest.fixture
def keyword_extractor():
    """
    Fixture that provides a KeywordExtractionTool instance with a mocked KeyBERT extractor.
    """
    extractor = KeywordExtractionTool()
    # Mock the KeyBERT model to avoid loading the actual model in a unit test
    extractor.keybert_extractor = MagicMock()
    extractor.keybert_extractor.extract_keywords.return_value = [("mocked keyword", 0.9)]
    return extractor

@pytest.mark.parametrize("text, expected_keywords", [
    ("Renewable energy sources include solar, wind, and hydroelectric power.", ["renewable energy", "hydroelectric power", "energy sources"]),
    ("The cat sat on the mat.", ["cat", "mat"]),
    ("This is a test.", ["test"])
])
def test_keyword_extraction_yake(keyword_extractor, text, expected_keywords):
    """
    Tests the YAKE keyword extraction with various inputs.
    """
    keywords = keyword_extractor.extract_keywords_yake(text)
    keyword_texts = [kw[0].lower() for kw in keywords]
    
    assert len(keywords) > 0
    for expected in expected_keywords:
        assert any(expected in kw for kw in keyword_texts)

def test_keyword_extraction_yake_empty_string(keyword_extractor):
    """
    Tests that the YAKE extractor handles empty strings gracefully.
    """
    keywords = keyword_extractor.extract_keywords_yake("")
    assert len(keywords) == 0

def test_comprehensive_keyword_extraction(keyword_extractor):
    """
    Tests the comprehensive extraction method, ensuring it calls both YAKE and KeyBERT.
    """
    text = "This is a test of the comprehensive keyword extraction."
    results = keyword_extractor.extract_comprehensive_keywords(text)
    
    assert "yake" in results
    assert "keybert" in results
    assert len(results["yake"]) > 0
    assert len(results["keybert"]) > 0
    
    # Check that our mocked KeyBERT was called
    keyword_extractor.keybert_extractor.extract_keywords.assert_called_once()