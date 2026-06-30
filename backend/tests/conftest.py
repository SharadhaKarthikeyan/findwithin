import pytest
import asyncio

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    policy = asyncio.get_event_loop_policy()
    res_loop = policy.new_event_loop()
    yield res_loop
    res_loop.close()
