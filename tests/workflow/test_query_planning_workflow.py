
import pytest
from unittest.mock import Mock, AsyncMock

from llama_index.core.workflow import Workflow
from src.workflows.query_planning_workflow import QueryPlanningWorkflow, SubQueriesExecutedEvent


@pytest.fixture
def mock_llm():
    # This mock will return a pre-defined list of sub-queries.
    llm = Mock()
    llm.acomplete = AsyncMock(return_value="1. What is X?\n2. How does Y work?")
    return llm


@pytest.fixture
def mock_query_engine():
    # This mock simulates a query engine that returns a simple result.
    engine = Mock()
    engine.aquery = AsyncMock(return_value="This is the answer.")
    return engine


@pytest.mark.asyncio
async def test_query_planning_workflow_e2e(mock_llm, mock_query_engine):
    # Arrange
    query_engines = {"default": mock_query_engine}
    workflow = QueryPlanningWorkflow(llm=mock_llm, query_engines=query_engines)

    # Act
    result = await workflow.run(query="Tell me about X and Y.")

    # Assert
    # Check that the sub-queries were extracted correctly.
    assert mock_llm.acomplete.call_count == 2
    # Check that the query engine was called for each sub-query.
    assert mock_query_engine.aquery.call_count == 2
    # Check that the final result is a string.
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_plan_query_step(mock_llm):
    # Arrange
    workflow = QueryPlanningWorkflow(llm=mock_llm, query_engines={})
    mock_context = Mock()
    mock_context.store = AsyncMock()
    mock_start_event = Mock(query="test query")

    # Act
    result_event = await workflow.plan_query(mock_context, mock_start_event)

    # Assert
    assert result_event.sub_queries == ["What is X?", "How does Y work?"]
    mock_context.store.set.assert_called_once_with("original_query", "test query")


@pytest.mark.asyncio
async def test_execute_sub_queries_step(mock_query_engine):
    # Arrange
    workflow = QueryPlanningWorkflow(llm=Mock(), query_engines={"default": mock_query_engine})
    mock_context = Mock()
    mock_decomposition_event = Mock(sub_queries=["sub_query_1", "sub_query_2"])

    # Act
    result_event = await workflow.execute_sub_queries(mock_context, mock_decomposition_event)

    # Assert
    assert isinstance(result_event, SubQueriesExecutedEvent)
    assert len(result_event.sub_results) == 2
    assert result_event.sub_results[0]["query"] == "sub_query_1"
    assert result_event.sub_results[1]["result"] == "This is the answer."
    assert mock_query_engine.aquery.call_count == 2
