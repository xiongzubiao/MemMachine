from unittest.mock import AsyncMock, MagicMock

from unittest.mock import MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine.common.configuration.database_conf import DatabasesConf, SqlAlchemyConf
from memmachine.common.resource_manager.database_manager import DatabaseManager
from memmachine.common.vector_graph_store import VectorGraphStore


@pytest.fixture
def mock_conf():
    """Mock StoragesConf with dummy connection configurations."""
    conf = MagicMock(spec=DatabasesConf)
    conf.relational_db_confs = {
        "sqlite1": SqlAlchemyConf(
            dialect="sqlite",
            driver="aiosqlite",
            path="test.db",
        ),
    }
    return conf


@pytest.mark.asyncio
async def test_get_vector_graph_store_from_sqlite(mock_conf):
    builder = DatabaseManager(mock_conf)
    store = await builder.get_vector_graph_store("sqlite1")
    assert isinstance(store, VectorGraphStore)


@pytest.mark.asyncio
async def test_build_sqlite(mock_conf):
    builder = DatabaseManager(mock_conf)
    await builder._build_sql_engines()

    assert "sqlite1" in builder.sql_engines
    assert isinstance(builder.sql_engines["sqlite1"], AsyncEngine)


@pytest.mark.asyncio
async def test_build_and_validate_sqlite():
    conf = MagicMock(spec=DatabasesConf)
    conf.relational_db_confs = {
        "sqlite1": SqlAlchemyConf(
            dialect="sqlite",
            driver="aiosqlite",
            path=":memory:",
        )
    }
    builder = DatabaseManager(conf)
    await builder.build_all(validate=True)
    # If no exception is raised, validation passed
    assert "sqlite1" in builder.sql_engines
    await builder.close()
    assert "sqlite1" not in builder.sql_engines


@pytest.mark.asyncio
async def test_build_all_without_validation(mock_conf):
    builder = DatabaseManager(mock_conf)
    await builder.build_all(validate=False)

    assert "sqlite1" in builder.sql_engines
