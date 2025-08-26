import pytest
from src.memory.short_term_memory import ShortTermMemory

@pytest.mark.asyncio
async def test_short_term_memory():
    """
    Tests the basic functionality of the ShortTermMemory class,
    ensuring that messages can be added and retrieved correctly.
    """
    memory = ShortTermMemory(session_id="test_short_term_session")
    
    # Add a message to the memory
    await memory.add_message("user", "Hello, I'm researching renewable energy")
    
    # Retrieve the context
    context = await memory.get_context()
    
    # Assertions
    assert len(context) == 1
    assert context[0].role == "user"
    assert context[0].content == "Hello, I'm researching renewable energy"
    
    # Test clearing the context
    await memory.clear_context()
    context_after_clear = await memory.get_context()
    assert len(context_after_clear) == 0