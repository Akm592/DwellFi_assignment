import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.workflows.main_workflow import MainResearchWorkflow
from llama_index.core.workflow import StartEvent


@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.acomplete = AsyncMock(return_value="Final response.")
    return llm


@pytest.fixture
def mock_memory_system():
    memory_system = {
        "short_term": AsyncMock(),
        "long_term": AsyncMock(),
    }
    memory_system["short_term"].get_context.return_value = []
    memory_system["long_term"].get_relevant_context.return_value = ""
    return memory_system


@pytest.fixture
def mock_query_engine():
    engine = Mock()
    engine.aquery = AsyncMock(return_value="Direct query result.")
    return engine


@pytest.fixture
def mock_query_planning_workflow():
    workflow = Mock()
    workflow.run = AsyncMock(return_value="Planned query result.")
    return workflow


@pytest.mark.asyncio
async def test_main_research_workflow_simple_query_e2e(
    mock_llm,
    mock_memory_system,
    mock_query_engine,
    mock_query_planning_workflow,
):
    # Arrange
    workflow = MainResearchWorkflow(
        llm=mock_llm,
        tools=[],
        memory_system=mock_memory_system,
        query_engines={"default": mock_query_engine},
        query_planning_workflow=mock_query_planning_workflow,
    )

    # Act
    result = await workflow.run(query="Simple query?")

    # Assert
    assert result["response"] == "Final response."
    assert mock_query_engine.aquery.call_count == 1
    assert mock_query_planning_workflow.run.call_count == 0
    assert mock_memory_system["short_term"].add_message.call_count == 2


@pytest.mark.asyncio
async def test_main_research_workflow_complex_query_e2e(
    mock_llm,
    mock_memory_system,
    mock_query_engine,
    mock_query_planning_workflow,
):
    # Arrange
    workflow = MainResearchWorkflow(
        llm=mock_llm,
        tools=[],
        memory_system=mock_memory_system,
        query_engines={"default": mock_query_engine},
        query_planning_workflow=mock_query_planning_workflow,
    )

    # Act
    # This query is designed to trigger the complexity threshold.
    result = await workflow.run(query="Compare and contrast X and Y and how they relate to Z?")

    # Assert
    assert result["response"] == "Final response."
    assert mock_query_engine.aquery.call_count == 0
    assert mock_query_planning_workflow.run.call_count == 1
    assert mock_memory_system["short_term"].add_message.call_count == 2
