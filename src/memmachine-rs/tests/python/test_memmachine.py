import asyncio
import tempfile
from pathlib import Path

import pytest

pytest_plugins = ["pytest_asyncio"]


@pytest.fixture
def temp_db_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def memmachine(temp_db_path):
    from memmachine_rust import MemMachine

    return MemMachine(str(temp_db_path))


@pytest.mark.asyncio
async def test_create_session(memmachine):
    from memmachine_rust import MemMachine

    info = await memmachine.create_session("test-session", "Test description")

    assert info.session_key == "test-session"
    assert info.description == "Test description"


@pytest.mark.asyncio
async def test_get_session(memmachine):
    await memmachine.create_session("test-session", "Test description")

    session = await memmachine.get_session("test-session")

    assert session is not None
    assert session.session_key == "test-session"


@pytest.mark.asyncio
async def test_get_nonexistent_session(memmachine):
    session = await memmachine.get_session("nonexistent")
    assert session is None


@pytest.mark.asyncio
async def test_add_episodes(memmachine):
    from memmachine_rust import EpisodeEntry, EpisodeRole, MemoryType, SessionData

    await memmachine.create_session("test-session", "Test")

    session_data = SessionData("test-session")
    entries = [
        EpisodeEntry(EpisodeRole.User, "Hello, how are you?"),
        EpisodeEntry(EpisodeRole.Assistant, "I'm doing well, thank you!"),
    ]

    ids = await memmachine.add_episodes(session_data, entries, [MemoryType.Episodic])

    assert len(ids) == 2
    assert all(id.startswith("ep-") for id in ids)


@pytest.mark.asyncio
async def test_episodes_count(memmachine):
    from memmachine_rust import EpisodeEntry, EpisodeRole, MemoryType, SessionData

    await memmachine.create_session("test-session", "Test")

    session_data = SessionData("test-session")
    entries = [
        EpisodeEntry(EpisodeRole.User, "Message 1"),
        EpisodeEntry(EpisodeRole.Assistant, "Message 2"),
        EpisodeEntry(EpisodeRole.User, "Message 3"),
    ]

    await memmachine.add_episodes(session_data, entries, [MemoryType.Episodic])

    count = await memmachine.episodes_count(session_data)

    assert count == 3


@pytest.mark.asyncio
async def test_delete_episodes(memmachine):
    from memmachine_rust import EpisodeEntry, EpisodeRole, MemoryType, SessionData

    await memmachine.create_session("test-session", "Test")

    session_data = SessionData("test-session")
    entries = [
        EpisodeEntry(EpisodeRole.User, "Message 1"),
        EpisodeEntry(EpisodeRole.Assistant, "Message 2"),
    ]

    ids = await memmachine.add_episodes(session_data, entries, [MemoryType.Episodic])

    deleted = await memmachine.delete_episodes([ids[0]], session_data)
    assert deleted == 1

    count = await memmachine.episodes_count(session_data)
    assert count == 1


@pytest.mark.asyncio
async def test_start_stop(memmachine):
    assert not memmachine.is_started()

    await memmachine.start()
    assert memmachine.is_started()

    await memmachine.stop()
    assert not memmachine.is_started()


@pytest.mark.asyncio
async def test_list_search(memmachine):
    from memmachine_rust import EpisodeEntry, EpisodeRole, MemoryType, SessionData

    await memmachine.create_session("test-session", "Test")

    session_data = SessionData("test-session")
    entries = [
        EpisodeEntry(EpisodeRole.User, "First message"),
        EpisodeEntry(EpisodeRole.Assistant, "Second message"),
        EpisodeEntry(EpisodeRole.User, "Third message"),
    ]

    await memmachine.add_episodes(session_data, entries, [MemoryType.Episodic])

    results = await memmachine.list_search(
        session_data, [MemoryType.Episodic], page_size=2, page_num=0
    )

    assert results.episodic_memory is not None
    assert len(results.episodic_memory) == 2


@pytest.mark.asyncio
async def test_delete_session(memmachine):
    from memmachine_rust import EpisodeEntry, EpisodeRole, MemoryType, SessionData

    await memmachine.create_session("test-session", "Test")

    session_data = SessionData("test-session")
    entries = [EpisodeEntry(EpisodeRole.User, "Message")]

    await memmachine.add_episodes(session_data, entries, [MemoryType.Episodic])

    await memmachine.delete_session(session_data)

    session = await memmachine.get_session("test-session")
    assert session is None


@pytest.mark.asyncio
async def test_search_sessions(memmachine):
    await memmachine.create_session("session-1", "First session")
    await memmachine.create_session("session-2", "Second session")

    sessions = await memmachine.search_sessions()

    assert len(sessions) == 2
    assert "session-1" in sessions
    assert "session-2" in sessions


@pytest.mark.asyncio
async def test_episode_entry_with_metadata(memmachine):
    from memmachine_rust import EpisodeEntry, EpisodeRole, MemoryType, SessionData

    await memmachine.create_session("test-session", "Test")

    session_data = SessionData("test-session")
    entries = [
        EpisodeEntry(
            EpisodeRole.User,
            "Hello",
            name="user1",
            metadata={"key": "value", "number": 42},
        ),
    ]

    ids = await memmachine.add_episodes(session_data, entries, [MemoryType.Episodic])

    assert len(ids) == 1

    results = await memmachine.list_search(session_data, [MemoryType.Episodic])
    episodes = results.episodic_memory

    assert episodes is not None
    assert len(episodes) == 1
    assert episodes[0].name == "user1"
    assert episodes[0].metadata == {"key": "value", "number": 42}
