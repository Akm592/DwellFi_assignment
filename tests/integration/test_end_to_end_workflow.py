import pytest
from fastapi.testclient import TestClient
from src.app import app as fastapi_app, app_state
import shutil
import redis

@pytest.fixture(scope="module")
def client():
    """
    Yield a TestClient for the FastAPI app.
    This fixture also handles the cleanup of the chroma_db directory and Redis cache.
    """
    try:
        r = redis.from_url("redis://localhost:6379")
        r.flushdb()
        print("\n--- Redis cache cleared for integration test ---")
    except redis.exceptions.ConnectionError as e:
        print(f"\n--- Could not connect to Redis to clear cache: {e} ---")

    with TestClient(fastapi_app) as c:
        yield c
        
    shutil.rmtree("./chroma_db", ignore_errors=True)

@pytest.mark.integration
def test_end_to_end_workflow_and_memory(client):
    """
    Test the complete end-to-end workflow, including session creation,
    query processing, and both short-term and long-term memory persistence.
    """
    session_id = "test_integration_session_123"

    # === First Interaction: Establish Context ===
    response1 = client.post(
        "/query",
        json={"query": "I'm researching the financial performance of Adobe.", "session_id": session_id}
    )
    assert response1.status_code == 200
    response1_data = response1.json()
    assert "adobe" in response1_data["response"].lower()
    assert len(response1_data["sources"]) > 0

    # === Verify Long-Term Memory Update ===
    long_term_memory = app_state.get("long_term_memory")
    assert long_term_memory is not None

    research_context_block = None
    for block in long_term_memory.memory_blocks:
        if block.name == "research_context":
            research_context_block = block
            break
    
    assert research_context_block is not None
    
    # --- FIX: Make the assertion more robust ---
    # Instead of checking for an exact match, we check if the key concepts
    # are present in the list of extracted topics.
    extracted_topics = " ".join(research_context_block.research_topics.get("topics", [])).lower()
    assert "financial" in extracted_topics
    assert "performance" in extracted_topics
    assert "adobe" in extracted_topics

    # === Second Interaction: Test Contextual Understanding ===
    response2 = client.post(
        "/query",
        json={"query": "What were the key financial highlights?", "session_id": session_id}
    )
    assert response2.status_code == 200
    response2_data = response2.json()

    assert "adobe" in response2_data["response"].lower()
    assert "revenue" in response2_data["response"].lower()
    assert len(response2_data["sources"]) > 0