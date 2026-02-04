"""Manage SQL engines and sqlite-backed graph stores."""

import asyncio
import logging
from asyncio import Lock
from typing import Any, Self
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from memmachine.common.configuration.database_conf import DatabasesConf
from memmachine.common.errors import SQLConfigurationError
from memmachine.common.vector_graph_store import VectorGraphStore
from memmachine.common.vector_graph_store.sqlite_vector_graph_store import (
    SqliteVectorGraphStore,
    SqliteVectorGraphStoreParams,
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Create and manage database backends with lazy initialization."""

    def __init__(self, conf: DatabasesConf) -> None:
        """Initialize with database configuration."""
        self.conf = conf
        self.graph_stores: dict[str, VectorGraphStore] = {}
        self.sql_engines: dict[str, AsyncEngine] = {}
        self._lock = Lock()
        self._sql_locks: dict[str, Lock] = {}

    async def build_all(self, validate: bool = False) -> Self:
        """Optionally eagerly initialize all backends."""
        relation_db_tasks = [
            self.async_get_sql_engine(name, validate=validate)
            for name in self.conf.relational_db_confs
        ]
        # Lazy build will occur in get_* calls, but build_all can trigger them
        await asyncio.gather(*relation_db_tasks)

        if validate:
            await self._validate_sql_engines()

        return self

    async def close(self) -> None:
        """Close all database connections."""
        async with self._lock:
            tasks = []
            for name, store in self.graph_stores.items():
                tasks.append(self._close_vector_store(name, store))
            for name, engine in self.sql_engines.items():
                tasks.append(self._close_async_engine(name, engine))
            await asyncio.gather(*tasks)
            self.graph_stores.clear()
            self.sql_engines.clear()
            self._sql_locks.clear()

    @staticmethod
    async def _close_async_engine(name: str, engine: AsyncEngine) -> None:
        try:
            await engine.dispose()
        except Exception as ex:
            logger.warning("Error disposing SQL engine '%s': %s", name, ex)

    @staticmethod
    async def _close_vector_store(name: str, store: VectorGraphStore) -> None:
        try:
            await store.close()
        except Exception as ex:
            logger.warning("Error closing vector graph store '%s': %s", name, ex)

    async def get_vector_graph_store(self, name: str) -> VectorGraphStore:
        """Return a sqlite-backed vector graph store by name."""
        if name in self.graph_stores:
            return self.graph_stores[name]
        conf = self.conf.relational_db_confs.get(name)
        if not conf:
            raise ValueError(f"Vector graph store config '{name}' not found.")
        if conf.dialect != "sqlite":
            raise ValueError(
                f"Vector graph store '{name}' must use sqlite, got '{conf.dialect}'."
            )
        engine = await self.async_get_sql_engine(name, validate=True)
        params = SqliteVectorGraphStoreParams(engine=engine)
        store = SqliteVectorGraphStore(params)
        self.graph_stores[name] = store
        return store

    # --- SQL ---

    async def _build_sql_engines(self) -> None:
        """
        Eagerly build all SQL engines.

        This simply calls the lazy initializer for each configured relational DB.
        """
        tasks = [
            self.async_get_sql_engine(name) for name in self.conf.relational_db_confs
        ]
        if tasks:
            await asyncio.gather(*tasks)

    async def async_get_sql_engine(
        self, name: str, validate: bool = False
    ) -> AsyncEngine:
        """Return a SQL engine, creating it if necessary (lazy)."""
        if name not in self._sql_locks:
            async with self._lock:
                self._sql_locks.setdefault(name, Lock())

        async with self._sql_locks[name]:
            if name in self.sql_engines:
                return self.sql_engines[name]

            conf = self.conf.relational_db_confs.get(name)
            if not conf:
                raise ValueError(f"SQL config '{name}' not found.")

            engine_kwargs: dict[str, Any] = {
                "echo": False,
                "future": True,
            }
            if conf.pool_size is not None:
                engine_kwargs["pool_size"] = conf.pool_size
            if conf.max_overflow is not None:
                engine_kwargs["max_overflow"] = conf.max_overflow

            engine = create_async_engine(conf.uri, **engine_kwargs)
            if validate:
                await self.validate_sql_engine(name, engine)
            self.sql_engines[name] = engine
            return engine

    def get_sql_engine(self, name: str) -> AsyncEngine:
        """Sync wrapper to get SQL engine lazily."""
        return asyncio.run(self.async_get_sql_engine(name, validate=True))

    @staticmethod
    async def validate_sql_engine(name: str, engine: AsyncEngine) -> None:
        """Validate connectivity for a single SQL engine."""
        try:
            logger.info("Validating SQL engine '%s'", name)
            async with engine.connect() as conn:
                result = await conn.execute(text("SELECT 1;"))
                row = result.fetchone()
            logger.info("SQL engine '%s' validated successfully", name)
        except Exception as e:
            raise SQLConfigurationError(
                f"SQL config '{name}' failed verification: {e}",
            ) from e

        if not row or row[0] != 1:
            raise SQLConfigurationError(
                f"Verification failed for SQL config '{name}'",
            )

    async def _validate_sql_engines(self) -> None:
        """Validate connectivity for each SQL engine."""
        for name, engine in self.sql_engines.items():
            await self.validate_sql_engine(name, engine)
