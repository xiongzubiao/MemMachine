"""Chroma-backed semantic storage implementation."""

from __future__ import annotations

import asyncio
import importlib
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.episode_store import EpisodeIdT
from memmachine.common.errors import InvalidArgumentError
from memmachine.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine.semantic_memory.semantic_model import (
    FeatureIdT,
    SemanticFeature,
    SetIdT,
)
from memmachine.semantic_memory.storage.storage_base import SemanticStorage


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float | None:
    if lhs.shape != rhs.shape:
        return None
    lnorm = float(np.linalg.norm(lhs))
    rnorm = float(np.linalg.norm(rhs))
    if lnorm == 0 or rnorm == 0:
        return None
    return float(np.dot(lhs, rhs) / (lnorm * rnorm))


class _NullCollection:
    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        return None

    def get(self, **_: Any) -> dict[str, list[Any]]:
        return {"ids": [], "embeddings": [], "metadatas": []}

    def delete(self, *, ids: list[str]) -> None:
        return None


@dataclass
class _FeatureEntry:
    id: FeatureIdT
    set_id: str
    semantic_type_id: str
    tag: str
    feature: str
    value: str
    embedding: np.ndarray
    metadata: dict[str, Any] | None = None
    citations: list[EpisodeIdT] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)


class ChromaSemanticStorageParams(BaseModel):
    """Parameters for ChromaSemanticStorage."""

    path: str = Field(..., description="Chroma persistence directory")
    collection_prefix: str = Field(
        default="memmachine",
        description="Prefix for Chroma collection names",
    )


class ChromaSemanticStorage(SemanticStorage):
    """Chroma-based semantic storage backend."""

    def __init__(self, params: ChromaSemanticStorageParams) -> None:
        self._client: Any | None = None
        self._feature_collection: Any = _NullCollection()
        self._history_collection: Any = _NullCollection()
        try:
            chroma_module = importlib.import_module("chromadb")
        except ModuleNotFoundError:
            self._chroma_available = False
        else:
            self._chroma_available = True
            chroma = cast(Any, chroma_module)
            client = chroma.PersistentClient(path=params.path)
            self._client = client
            prefix = params.collection_prefix
            self._feature_collection = client.get_or_create_collection(
                name=f"{prefix}_semantic_features",
                metadata={"hnsw:space": "cosine"},
            )
            self._history_collection = client.get_or_create_collection(
                name=f"{prefix}_semantic_history",
                metadata={"hnsw:space": "cosine"},
            )

        self._features_by_id: dict[FeatureIdT, _FeatureEntry] = {}
        self._feature_ids_by_set: dict[str, list[FeatureIdT]] = {}
        self._feature_embedding_dimension: int | None = None
        self._set_history_map: dict[str, dict[EpisodeIdT, bool]] = {}
        self._history_created_at: dict[tuple[str, EpisodeIdT], datetime] = {}
        self._history_to_sets: dict[EpisodeIdT, dict[str, bool]] = {}
        self._lock = asyncio.Lock()

    async def startup(self) -> None:
        await self._load_features()
        await self._load_history()

    async def cleanup(self) -> None:
        return None

    async def delete_all(self) -> None:
        async with self._lock:
            await self._delete_all_collection_entries(self._feature_collection)
            await self._delete_all_collection_entries(self._history_collection)
            self._features_by_id.clear()
            self._feature_ids_by_set.clear()
            self._set_history_map.clear()
            self._history_created_at.clear()
            self._history_to_sets.clear()

    async def get_feature(
        self,
        feature_id: FeatureIdT,
        load_citations: bool = False,
    ) -> SemanticFeature | None:
        async with self._lock:
            feature_id = self._normalize_feature_id(feature_id)
            entry = self._features_by_id.get(feature_id)
            if entry is None:
                return None
            return self._feature_to_model(entry, load_citations=load_citations)

    async def add_feature(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
        feature: str,
        value: str,
        tag: str,
        embedding: InstanceOf[np.ndarray],
        metadata: dict[str, Any] | None = None,
    ) -> FeatureIdT:
        entry = _FeatureEntry(
            id=FeatureIdT(str(self._generate_feature_id())),
            set_id=set_id,
            semantic_type_id=category_name,
            tag=tag,
            feature=feature,
            value=value,
            embedding=np.array(embedding, dtype=float, copy=True),
            metadata=dict(metadata or {}) or None,
        )
        async with self._lock:
            await self._persist_feature(entry)
            self._features_by_id[entry.id] = entry
            self._feature_ids_by_set.setdefault(set_id, []).append(entry.id)
            return entry.id

    async def update_feature(
        self,
        feature_id: FeatureIdT,
        *,
        set_id: SetIdT | None = None,
        category_name: str | None = None,
        feature: str | None = None,
        value: str | None = None,
        tag: str | None = None,
        embedding: InstanceOf[np.ndarray] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        async with self._lock:
            feature_id = self._normalize_feature_id(feature_id)
            entry = self._features_by_id.get(feature_id)
            if entry is None:
                return

            self._handle_set_change(entry, feature_id, set_id)
            self._update_entry_fields(
                entry,
                category_name=category_name,
                feature=feature,
                value=value,
                tag=tag,
                embedding=embedding,
                metadata=metadata,
            )
            entry.updated_at = _utcnow()
            await self._persist_feature(entry)

    async def delete_features(self, feature_ids: list[FeatureIdT]) -> None:
        if not feature_ids:
            return

        async with self._lock:
            ids_to_delete: list[str] = []
            for feature_id in feature_ids:
                feature_id = self._normalize_feature_id(feature_id)
                entry = self._features_by_id.pop(feature_id, None)
                if entry is None:
                    continue
                self._remove_feature_from_index(entry)
                ids_to_delete.append(entry.id)
            if ids_to_delete:
                await asyncio.to_thread(
                    self._feature_collection.delete, ids=ids_to_delete
                )

    async def get_feature_set(
        self,
        *,
        filter_expr: FilterExpr | None = None,
        page_size: int | None = None,
        page_num: int | None = None,
        vector_search_opts: SemanticStorage.VectorSearchOpts | None = None,
        tag_threshold: int | None = None,
        load_citations: bool = False,
    ) -> list[SemanticFeature]:
        if page_num is not None:
            if page_size is None:
                raise InvalidArgumentError("Cannot specify offset without limit")
            if page_num < 0:
                raise InvalidArgumentError("Offset must be non-negative")

        fetch_limit = page_size
        if page_size is not None and page_num:
            fetch_limit = page_size * (page_num + 1)

        async with self._lock:
            entries = self._filter_features(
                set_ids=None,
                type_names=None,
                feature_names=None,
                tags=None,
                filter_expr=filter_expr,
                k=fetch_limit,
                vector_search_opts=vector_search_opts,
                tag_threshold=tag_threshold,
            )
            if page_size is not None:
                start_index = page_size * (page_num or 0)
                entries = entries[start_index : start_index + page_size]
            return [
                self._feature_to_model(entry, load_citations=load_citations)
                for entry in entries
            ]

    async def delete_feature_set(
        self,
        *,
        filter_expr: FilterExpr | None = None,
    ) -> None:
        async with self._lock:
            entries = self._filter_features(
                set_ids=None,
                type_names=None,
                feature_names=None,
                tags=None,
                filter_expr=filter_expr,
                k=None,
                vector_search_opts=None,
                tag_threshold=None,
            )
            ids = [entry.id for entry in entries]
        if ids:
            await self.delete_features(ids)

    async def add_citations(
        self,
        feature_id: FeatureIdT,
        history_ids: list[EpisodeIdT],
    ) -> None:
        if not history_ids:
            return
        async with self._lock:
            feature_id = self._normalize_feature_id(feature_id)
            entry = self._features_by_id.get(feature_id)
            if entry is None:
                return

            existing = set(entry.citations)
            for history_id in history_ids:
                history_id = EpisodeIdT(history_id)
                if history_id not in existing:
                    entry.citations.append(history_id)
                    existing.add(history_id)
            entry.updated_at = _utcnow()
            await self._persist_feature(entry)

    async def get_history_messages(
        self,
        *,
        set_ids: list[SetIdT] | None = None,
        limit: int | None = None,
        is_ingested: bool | None = None,
    ) -> list[EpisodeIdT]:
        async with self._lock:
            rows = self._history_rows_for_sets(set_ids)
            rows = self._filter_history_rows(rows, is_ingested)
            history_ids = [history_id for history_id, _ in rows]
            if limit is not None:
                history_ids = history_ids[:limit]
            return history_ids

    async def get_history_messages_count(
        self,
        *,
        set_ids: list[SetIdT] | None = None,
        is_ingested: bool | None = None,
    ) -> int:
        async with self._lock:
            rows = self._history_rows_for_sets(set_ids)
            filtered_rows = self._filter_history_rows(rows, is_ingested)
            return len(filtered_rows)

    async def add_history_to_set(self, set_id: SetIdT, history_id: EpisodeIdT) -> None:
        async with self._lock:
            history_map = self._set_history_map.setdefault(set_id, {})
            history_id = EpisodeIdT(history_id)
            is_new_association = history_id not in history_map
            history_map[history_id] = history_map.get(history_id, False)
            self._history_created_at[(set_id, history_id)] = _utcnow()
            self._history_to_sets.setdefault(history_id, {})[set_id] = history_map[
                history_id
            ]
            if is_new_association:
                await self._persist_history_entry(
                    set_id, history_id, history_map[history_id]
                )
            else:
                await self._persist_history_entry(
                    set_id, history_id, history_map[history_id]
                )

    async def delete_history(self, history_ids: list[EpisodeIdT]) -> None:
        if not history_ids:
            return
        async with self._lock:
            ids_to_delete: list[str] = []
            for history_id in history_ids:
                normalized_id = EpisodeIdT(history_id)
                referencing_sets = self._history_to_sets.pop(normalized_id, None)
                if not referencing_sets:
                    continue
                for set_id in list(referencing_sets.keys()):
                    history_map = self._set_history_map.get(set_id)
                    if history_map is None:
                        continue
                    history_map.pop(normalized_id, None)
                    self._history_created_at.pop((set_id, normalized_id), None)
                    ids_to_delete.append(self._history_record_id(set_id, normalized_id))
                    if not history_map:
                        self._set_history_map.pop(set_id, None)
            if ids_to_delete:
                await asyncio.to_thread(
                    self._history_collection.delete, ids=ids_to_delete
                )

    async def delete_history_set(self, set_ids: list[SetIdT]) -> None:
        if not set_ids:
            return
        async with self._lock:
            ids_to_delete: list[str] = []
            for set_id in set_ids:
                history_map = self._set_history_map.pop(set_id, None)
                if history_map is None:
                    continue
                for history_id in list(history_map.keys()):
                    self._history_created_at.pop((set_id, history_id), None)
                    sets_map = self._history_to_sets.get(history_id)
                    if sets_map:
                        sets_map.pop(set_id, None)
                        if not sets_map:
                            self._history_to_sets.pop(history_id, None)
                    ids_to_delete.append(self._history_record_id(set_id, history_id))
            if ids_to_delete:
                await asyncio.to_thread(
                    self._history_collection.delete, ids=ids_to_delete
                )

    async def mark_messages_ingested(
        self,
        *,
        set_id: SetIdT,
        history_ids: list[EpisodeIdT],
    ) -> None:
        if not history_ids:
            return
        async with self._lock:
            set_map = self._set_history_map.get(set_id)
            if not set_map:
                return
            for history_id in history_ids:
                history_id = EpisodeIdT(history_id)
                if history_id in set_map:
                    set_map[history_id] = True
                    self._history_to_sets.setdefault(history_id, {})[set_id] = True
                    await self._persist_history_entry(set_id, history_id, True)

    async def get_history_set_ids(
        self,
        *,
        min_uningested_messages: int | None = None,
        older_than: datetime | None = None,
    ) -> list[SetIdT]:
        async with self._lock:
            matching_sets: set[SetIdT] = set()
            filters_applied = False

            if min_uningested_messages is not None and min_uningested_messages > 0:
                filters_applied = True
                for set_id, history_map in self._set_history_map.items():
                    if not history_map:
                        continue
                    count_uningested = sum(
                        1 for ingested in history_map.values() if not ingested
                    )
                    if count_uningested >= min_uningested_messages:
                        matching_sets.add(set_id)

            if older_than is not None:
                filters_applied = True
                for set_id, history_map in self._set_history_map.items():
                    for history_id, ingested in history_map.items():
                        if ingested:
                            continue
                        created_at = self._history_created_at.get(
                            (set_id, history_id), _utcnow()
                        )
                        if created_at <= older_than:
                            matching_sets.add(set_id)
                            break

            if not filters_applied:
                matching_sets.update(self._set_history_map.keys())

            return list(matching_sets)

    def _history_rows_for_sets(
        self, set_ids: list[SetIdT] | None
    ) -> list[tuple[EpisodeIdT, bool]]:
        rows: list[tuple[EpisodeIdT, bool]] = []
        for set_id, history_map in self._set_history_map.items():
            if set_ids is not None and set_id not in set_ids:
                continue
            rows.extend(
                (history_id, ingested) for history_id, ingested in history_map.items()
            )
        return rows

    @staticmethod
    def _filter_history_rows(
        rows: list[tuple[EpisodeIdT, bool]],
        is_ingested: bool | None,
    ) -> list[tuple[EpisodeIdT, bool]]:
        if is_ingested is None:
            return rows
        return [row for row in rows if row[1] == is_ingested]

    def _filter_features(
        self,
        *,
        set_ids: list[str] | None,
        type_names: list[str] | None,
        feature_names: list[str] | None,
        tags: list[str] | None,
        filter_expr: FilterExpr | None,
        k: int | None,
        vector_search_opts: SemanticStorage.VectorSearchOpts | None,
        tag_threshold: int | None,
    ) -> list[_FeatureEntry]:
        entries = list(self._features_by_id.values())
        entries = self._apply_basic_filters(
            entries,
            set_ids=set_ids,
            type_names=type_names,
            feature_names=feature_names,
            tags=tags,
        )
        entries = self._apply_filter_expression(entries, filter_expr)
        entries = self._apply_vector_filter(entries, vector_search_opts)

        if k is not None:
            entries = entries[:k]

        if tag_threshold is not None:
            return self._apply_tag_threshold(entries, tag_threshold)

        return entries

    @staticmethod
    def _apply_basic_filters(
        entries: list[_FeatureEntry],
        *,
        set_ids: list[str] | None,
        type_names: list[str] | None,
        feature_names: list[str] | None,
        tags: list[str] | None,
    ) -> list[_FeatureEntry]:
        filters: list[tuple[list[str] | None, Callable[[_FeatureEntry], str]]] = [
            (set_ids, lambda entry: entry.set_id),
            (type_names, lambda entry: entry.semantic_type_id),
            (feature_names, lambda entry: entry.feature),
            (tags, lambda entry: entry.tag),
        ]

        filtered_entries = entries
        for allowed_values, getter in filters:
            if not allowed_values:
                continue
            allowed = set(allowed_values)
            filtered_entries = [
                entry for entry in filtered_entries if getter(entry) in allowed
            ]

        return filtered_entries

    def _apply_filter_expression(
        self,
        entries: list[_FeatureEntry],
        filter_expr: FilterExpr | None,
    ) -> list[_FeatureEntry]:
        if filter_expr is None:
            return entries
        return [
            entry for entry in entries if self._evaluate_filter_expr(entry, filter_expr)
        ]

    def _evaluate_filter_expr(
        self,
        entry: _FeatureEntry,
        expr: FilterExpr,
    ) -> bool:
        if isinstance(expr, FilterComparison):
            return self._evaluate_comparison(entry, expr)
        if isinstance(expr, FilterAnd):
            return self._evaluate_filter_expr(
                entry, expr.left
            ) and self._evaluate_filter_expr(entry, expr.right)
        if isinstance(expr, FilterOr):
            return self._evaluate_filter_expr(
                entry, expr.left
            ) or self._evaluate_filter_expr(entry, expr.right)
        raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")

    def _evaluate_comparison(
        self,
        entry: _FeatureEntry,
        comparison: FilterComparison,
    ) -> bool:
        value, is_metadata = self._resolve_entry_field(entry, comparison.field)
        match comparison.op:
            case "=":
                return self._compare_equals(value, comparison.value, is_metadata)
            case "in":
                return self._compare_in(value, comparison.value, is_metadata)
            case ">" | "<" | ">=" | "<=":
                return self._compare_order(
                    value,
                    comparison.value,
                    is_metadata,
                    comparison.op,
                )
            case "is_null":
                return value is None
            case "is_not_null":
                return value is not None
            case _:
                raise ValueError(f"Unsupported operator: {comparison.op}")

    def _compare_equals(self, value: Any, expected: Any, is_metadata: bool) -> bool:
        if isinstance(expected, list):
            raise TypeError("'=' comparison cannot accept list values")
        if is_metadata and expected is not None:
            expected = self._normalize_metadata_value(expected)
        if is_metadata and value is not None:
            value = self._normalize_metadata_value(value)
        return value == expected

    def _compare_in(self, value: Any, candidates: Any, is_metadata: bool) -> bool:
        if not isinstance(candidates, list):
            raise TypeError("IN comparison requires a list of values")
        if is_metadata:
            candidates = [
                self._normalize_metadata_value(v) if v is not None else None
                for v in candidates
            ]
            if value is not None:
                value = self._normalize_metadata_value(value)
        return value in candidates

    def _compare_order(
        self,
        value: Any,
        expected: Any,
        is_metadata: bool,
        op: str,
    ) -> bool:
        if isinstance(expected, list):
            raise TypeError(f"'{op}' comparison cannot accept list values")
        if value is None:
            return False
        if is_metadata:
            expected = self._normalize_metadata_value(expected)
            value = self._normalize_metadata_value(value)
        operations: dict[str, Callable[[Any, Any], bool]] = {
            ">": lambda lhs, rhs: lhs > rhs,
            "<": lambda lhs, rhs: lhs < rhs,
            ">=": lambda lhs, rhs: lhs >= rhs,
            "<=": lambda lhs, rhs: lhs <= rhs,
        }
        return operations[op](value, expected)

    def _resolve_entry_field(
        self,
        entry: _FeatureEntry,
        field: str,
    ) -> tuple[Any, bool]:
        field_mapping: dict[str, Any] = {
            "set_id": entry.set_id,
            "set": entry.set_id,
            "semantic_category_id": entry.semantic_type_id,
            "category_name": entry.semantic_type_id,
            "category": entry.semantic_type_id,
            "tag_id": entry.tag,
            "tag": entry.tag,
            "feature": entry.feature,
            "feature_name": entry.feature,
            "value": entry.value,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
        }
        if field in field_mapping:
            return field_mapping[field], False

        if field.startswith(("m.", "metadata.")):
            key = field.split(".", 1)[1]
            metadata = entry.metadata or {}
            return metadata.get(key), True

        return None, False

    @staticmethod
    def _normalize_metadata_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return ""
        return str(value)

    def _apply_vector_filter(
        self,
        entries: list[_FeatureEntry],
        vector_search_opts: SemanticStorage.VectorSearchOpts | None,
    ) -> list[_FeatureEntry]:
        if vector_search_opts is None:
            return sorted(entries, key=lambda e: (e.created_at, e.id))

        scored_entries: list[tuple[float, _FeatureEntry]] = []
        for entry in entries:
            similarity = self._resolve_similarity(
                entry.embedding,
                vector_search_opts.query_embedding,
            )
            if not self._passes_min_distance(
                similarity,
                vector_search_opts.min_distance,
            ):
                continue
            scored_entries.append((similarity, entry))

        scored_entries.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored_entries]

    @staticmethod
    def _apply_tag_threshold(
        entries: list[_FeatureEntry],
        tag_threshold: int,
    ) -> list[_FeatureEntry]:
        counts = Counter(entry.tag for entry in entries)
        return [entry for entry in entries if counts[entry.tag] >= tag_threshold]

    @staticmethod
    def _resolve_similarity(
        embedding: np.ndarray,
        query_embedding: np.ndarray,
    ) -> float:
        similarity = _cosine_similarity(embedding, query_embedding)
        return similarity if similarity is not None else float("-inf")

    @staticmethod
    def _passes_min_distance(similarity: float, min_distance: float | None) -> bool:
        if min_distance is None:
            return True
        return similarity >= min_distance

    @staticmethod
    def _normalize_feature_id(feature_id: FeatureIdT) -> FeatureIdT:
        return FeatureIdT(str(feature_id))

    def _handle_set_change(
        self,
        entry: _FeatureEntry,
        feature_id: FeatureIdT,
        new_set_id: SetIdT | None,
    ) -> None:
        if new_set_id is None or new_set_id == entry.set_id:
            return
        if entry.set_id in self._feature_ids_by_set:
            self._feature_ids_by_set[entry.set_id] = [
                feature_id
                for feature_id in self._feature_ids_by_set[entry.set_id]
                if feature_id != entry.id
            ]
            if not self._feature_ids_by_set[entry.set_id]:
                self._feature_ids_by_set.pop(entry.set_id, None)
        self._feature_ids_by_set.setdefault(new_set_id, []).append(entry.id)
        entry.set_id = new_set_id

    def _update_entry_fields(
        self,
        entry: _FeatureEntry,
        *,
        category_name: str | None,
        feature: str | None,
        value: str | None,
        tag: str | None,
        embedding: InstanceOf[np.ndarray] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        if category_name is not None:
            entry.semantic_type_id = category_name
        if feature is not None:
            entry.feature = feature
        if value is not None:
            entry.value = value
        if tag is not None:
            entry.tag = tag
        if embedding is not None:
            entry.embedding = np.array(embedding, dtype=float, copy=True)
        if metadata is not None:
            entry.metadata = dict(metadata)

    def _feature_to_model(
        self,
        entry: _FeatureEntry,
        *,
        load_citations: bool,
    ) -> SemanticFeature:
        metadata = SemanticFeature.Metadata(
            id=entry.id,
            citations=list(entry.citations) if load_citations else None,
            other=entry.metadata,
        )
        return SemanticFeature(
            set_id=entry.set_id,
            category=entry.semantic_type_id,
            tag=entry.tag,
            feature_name=entry.feature,
            value=entry.value,
            metadata=metadata,
        )

    def _remove_feature_from_index(self, entry: _FeatureEntry) -> None:
        if entry.set_id not in self._feature_ids_by_set:
            return
        self._feature_ids_by_set[entry.set_id] = [
            feature_id
            for feature_id in self._feature_ids_by_set[entry.set_id]
            if feature_id != entry.id
        ]
        if not self._feature_ids_by_set[entry.set_id]:
            self._feature_ids_by_set.pop(entry.set_id, None)

    def _generate_feature_id(self) -> str:
        return f"feature-{int(_utcnow().timestamp() * 1_000_000)}-{id(self)}"

    def _serialize_feature_metadata(self, entry: _FeatureEntry) -> dict[str, Any]:
        return {
            "set_id": entry.set_id,
            "category": entry.semantic_type_id,
            "tag": entry.tag,
            "feature": entry.feature,
            "value": entry.value,
            "embedding_json": json.dumps(entry.embedding.tolist()),
            "metadata_json": json.dumps(entry.metadata or {}),
            "citations_json": json.dumps(entry.citations),
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
        }

    @staticmethod
    def _deserialize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        payload = metadata.get("metadata_json")
        if not payload:
            return {}
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _deserialize_citations(metadata: dict[str, Any]) -> list[EpisodeIdT]:
        payload = metadata.get("citations_json")
        if not payload:
            return []
        try:
            return [EpisodeIdT(str(val)) for val in json.loads(payload)]
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime:
        if not value:
            return _utcnow()
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return _utcnow()

    def _resolve_feature_embedding_dimension(self, embedding: np.ndarray) -> int:
        if self._feature_embedding_dimension is None:
            self._feature_embedding_dimension = int(embedding.shape[0])
        return self._feature_embedding_dimension

    async def _persist_feature(self, entry: _FeatureEntry) -> None:
        if self._feature_collection is None:
            return
        metadata = self._serialize_feature_metadata(entry)
        embedding_dimension = self._resolve_feature_embedding_dimension(entry.embedding)
        placeholder_embedding = np.zeros(embedding_dimension, dtype=float).tolist()
        await asyncio.to_thread(
            self._feature_collection.upsert,
            ids=[entry.id],
            embeddings=[placeholder_embedding],
            metadatas=[metadata],
            documents=[entry.value],
        )

    async def _persist_history_entry(
        self, set_id: SetIdT, history_id: EpisodeIdT, ingested: bool
    ) -> None:
        if self._history_collection is None:
            return
        record_id = self._history_record_id(set_id, history_id)
        metadata = {
            "set_id": set_id,
            "history_id": str(history_id),
            "ingested": ingested,
            "created_at": self._history_created_at.get(
                (set_id, history_id), _utcnow()
            ).isoformat(),
        }
        await asyncio.to_thread(
            self._history_collection.upsert,
            ids=[record_id],
            embeddings=[[0.0]],
            metadatas=[metadata],
            documents=[""],
        )

    @staticmethod
    def _history_record_id(set_id: SetIdT, history_id: EpisodeIdT) -> str:
        return f"{set_id}:{history_id}"

    async def _delete_all_collection_entries(self, collection: Any) -> None:
        if collection is None:
            return
        result = await asyncio.to_thread(collection.get)
        ids = result.get("ids", []) if isinstance(result, dict) else []
        if ids:
            await asyncio.to_thread(collection.delete, ids=ids)

    async def _load_features(self) -> None:
        if self._feature_collection is None:
            return
        result = await asyncio.to_thread(
            self._feature_collection.get,
            include=["embeddings", "metadatas"],
        )
        if not isinstance(result, dict):
            return
        ids = result.get("ids", [])
        embeddings = result.get("embeddings", [])
        metadatas = result.get("metadatas", [])

        for feature_id, embedding, metadata in zip(
            ids, embeddings, metadatas, strict=False
        ):
            metadata = metadata or {}
            if self._feature_embedding_dimension is None and embedding is not None:
                try:
                    self._feature_embedding_dimension = len(embedding)
                except TypeError:
                    self._feature_embedding_dimension = None
            embedding_payload = metadata.get("embedding_json")
            if embedding_payload:
                try:
                    embedding = json.loads(embedding_payload)
                except json.JSONDecodeError:
                    pass
            entry = _FeatureEntry(
                id=FeatureIdT(str(feature_id)),
                set_id=str(metadata.get("set_id", "")),
                semantic_type_id=str(metadata.get("category", "")),
                tag=str(metadata.get("tag", "")),
                feature=str(metadata.get("feature", "")),
                value=str(metadata.get("value", "")),
                embedding=np.array(embedding, dtype=float),
                metadata=self._deserialize_metadata(metadata),
                citations=self._deserialize_citations(metadata),
                created_at=self._parse_datetime(metadata.get("created_at")),
                updated_at=self._parse_datetime(metadata.get("updated_at")),
            )
            self._features_by_id[entry.id] = entry
            self._feature_ids_by_set.setdefault(entry.set_id, []).append(entry.id)

    async def _load_history(self) -> None:
        if self._history_collection is None:
            return
        result = await asyncio.to_thread(
            self._history_collection.get,
            include=["metadatas"],
        )
        if not isinstance(result, dict):
            return
        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [])

        for record_id, metadata in zip(ids, metadatas, strict=False):
            metadata = metadata or {}
            set_id = metadata.get("set_id")
            history_id = metadata.get("history_id")
            if not set_id or not history_id:
                continue
            ingested = bool(metadata.get("ingested", False))
            set_id = str(set_id)
            history_id = EpisodeIdT(str(history_id))

            self._set_history_map.setdefault(set_id, {})[history_id] = ingested
            self._history_to_sets.setdefault(history_id, {})[set_id] = ingested
            created_at = self._parse_datetime(metadata.get("created_at"))
            self._history_created_at[(set_id, history_id)] = created_at
