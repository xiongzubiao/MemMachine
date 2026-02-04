"""SQLite-backed vector graph store implementation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import cmp_to_key
from math import sqrt

from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine.common.data_types import FilterablePropertyValue, SimilarityMetric
from memmachine.common.filter.filter_parser import And, Comparison, FilterExpr, Or

from .data_types import Edge, Node, OrderedPropertyValue, PropertyValue
from .vector_graph_store import VectorGraphStore


@dataclass(frozen=True)
class _EdgeRecord:
    relation: str
    source_collection: str
    target_collection: str
    edge: Edge


@dataclass(frozen=True)
class SqliteVectorGraphStoreParams:
    engine: AsyncEngine


class SqliteVectorGraphStore(VectorGraphStore):
    """In-memory vector graph store with SQLite-compatible configuration."""

    def __init__(self, params: SqliteVectorGraphStoreParams) -> None:
        self._engine = params.engine
        self._nodes: dict[str, dict[str, Node]] = defaultdict(dict)
        self._edges: list[_EdgeRecord] = []

    async def add_nodes(
        self,
        *,
        collection: str,
        nodes: Iterable[Node],
    ) -> None:
        collection_nodes = self._nodes[collection]
        for node in nodes:
            collection_nodes[node.uid] = node

    async def add_edges(
        self,
        *,
        relation: str,
        source_collection: str,
        target_collection: str,
        edges: Iterable[Edge],
    ) -> None:
        for edge in edges:
            self._edges.append(
                _EdgeRecord(
                    relation=relation,
                    source_collection=source_collection,
                    target_collection=target_collection,
                    edge=edge,
                )
            )

    async def search_similar_nodes(
        self,
        *,
        collection: str,
        embedding_name: str,
        query_embedding: list[float],
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        limit: int | None = 100,
        property_filter: FilterExpr | None = None,
    ) -> list[Node]:
        candidates: list[tuple[float, Node]] = []
        for node in self._nodes.get(collection, {}).values():
            if not _matches_filter(node.properties, property_filter):
                continue
            embedding = node.embeddings.get(embedding_name)
            if embedding is None:
                continue
            score = _similarity_score(
                query_embedding,
                embedding[0],
                similarity_metric,
            )
            candidates.append((score, node))

        reverse = similarity_metric in {SimilarityMetric.COSINE, SimilarityMetric.DOT}
        candidates.sort(key=lambda item: item[0], reverse=reverse)
        nodes = [node for _, node in candidates]
        if limit is None:
            return nodes
        return nodes[:limit]

    async def search_related_nodes(
        self,
        *,
        relation: str,
        other_collection: str,
        this_collection: str,
        this_node_uid: str,
        find_sources: bool = True,
        find_targets: bool = True,
        limit: int | None = None,
        edge_property_filter: FilterExpr | None = None,
        node_property_filter: FilterExpr | None = None,
    ) -> list[Node]:
        results: list[Node] = []
        seen_uids: set[str] = set()

        def add_node(node: Node | None) -> None:
            if node is None:
                return
            if node.uid in seen_uids:
                return
            seen_uids.add(node.uid)
            results.append(node)

        for record in self._edges:
            if record.relation != relation:
                continue
            if not _matches_filter(record.edge.properties, edge_property_filter):
                continue

            if (
                find_sources
                and record.target_collection == this_collection
                and record.source_collection == other_collection
                and record.edge.target_uid == this_node_uid
            ):
                node = self._nodes.get(other_collection, {}).get(record.edge.source_uid)
                if node and _matches_filter(node.properties, node_property_filter):
                    add_node(node)

            if (
                find_targets
                and record.source_collection == this_collection
                and record.target_collection == other_collection
                and record.edge.source_uid == this_node_uid
            ):
                node = self._nodes.get(other_collection, {}).get(record.edge.target_uid)
                if node and _matches_filter(node.properties, node_property_filter):
                    add_node(node)

            if limit is not None and len(results) >= limit:
                return results[:limit]

        return results

    async def search_directional_nodes(
        self,
        *,
        collection: str,
        by_properties: Iterable[str],
        starting_at: Iterable[OrderedPropertyValue | None],
        order_ascending: Iterable[bool],
        include_equal_start: bool = False,
        limit: int | None = 1,
        property_filter: FilterExpr | None = None,
    ) -> list[Node]:
        properties = list(by_properties)
        start_values = list(starting_at)
        directions = list(order_ascending)

        nodes = [
            node
            for node in self._nodes.get(collection, {}).values()
            if _matches_filter(node.properties, property_filter)
        ]

        def compare_nodes(left: Node, right: Node) -> int:
            for prop, asc in zip(properties, directions, strict=False):
                left_val = _as_ordered_value(left.properties.get(prop))
                right_val = _as_ordered_value(right.properties.get(prop))
                cmp_val = _compare_values(left_val, right_val)
                if cmp_val != 0:
                    return cmp_val if asc else -cmp_val
            return 0

        nodes.sort(key=cmp_to_key(compare_nodes))

        filtered: list[Node] = []
        for node in nodes:
            if not _passes_starting_at(
                node,
                properties,
                start_values,
                directions,
                include_equal_start,
            ):
                continue
            filtered.append(node)
            if limit is not None and len(filtered) >= limit:
                return filtered[:limit]

        return filtered

    async def search_matching_nodes(
        self,
        *,
        collection: str,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[Node]:
        nodes = [
            node
            for node in self._nodes.get(collection, {}).values()
            if _matches_filter(node.properties, property_filter)
        ]
        if limit is None:
            return nodes
        return nodes[:limit]

    async def get_nodes(
        self,
        *,
        collection: str,
        node_uids: Iterable[str],
    ) -> list[Node]:
        collection_nodes = self._nodes.get(collection, {})
        return [node for uid in node_uids if (node := collection_nodes.get(uid))]

    async def delete_nodes(
        self,
        *,
        collection: str,
        node_uids: Iterable[str],
    ) -> None:
        collection_nodes = self._nodes.get(collection, {})
        delete_ids = set(node_uids)
        for node_id in delete_ids:
            collection_nodes.pop(node_id, None)

        if not delete_ids:
            return

        retained_edges: list[_EdgeRecord] = []
        for record in self._edges:
            if (
                record.source_collection == collection
                and record.edge.source_uid in delete_ids
            ):
                continue
            if (
                record.target_collection == collection
                and record.edge.target_uid in delete_ids
            ):
                continue
            retained_edges.append(record)
        self._edges = retained_edges

    async def delete_all_data(self) -> None:
        self._nodes.clear()
        self._edges.clear()

    async def close(self) -> None:
        return None


def _matches_filter(
    properties: Mapping[str, PropertyValue],
    expr: FilterExpr | None,
) -> bool:
    if expr is None:
        return True
    if isinstance(expr, Comparison):
        value = properties.get(expr.field)
        return _evaluate_comparison(value, expr)
    if isinstance(expr, And):
        return _matches_filter(properties, expr.left) and _matches_filter(
            properties, expr.right
        )
    if isinstance(expr, Or):
        return _matches_filter(properties, expr.left) or _matches_filter(
            properties, expr.right
        )
    return True


def _evaluate_comparison(value: PropertyValue, comp: Comparison) -> bool:
    if comp.op == "is_null":
        return value is None
    if comp.op == "is_not_null":
        return value is not None

    if comp.op == "=":
        return value == comp.value
    if comp.op == "in":
        if not isinstance(comp.value, list):
            return False
        return value in comp.value

    if value is None:
        return False

    if comp.op in {">", "<", ">=", "<="}:
        if isinstance(comp.value, list):
            return False
        result = _compare_filter_values(value, comp.value)
        if result is None:
            return False
        if comp.op == ">":
            return result > 0
        if comp.op == "<":
            return result < 0
        if comp.op == ">=":
            return result >= 0
        return result <= 0
    return False


def _similarity_score(
    left: list[float],
    right: list[float],
    metric: SimilarityMetric,
) -> float:
    if metric == SimilarityMetric.DOT:
        return sum(l * r for l, r in zip(left, right, strict=False))
    if metric == SimilarityMetric.COSINE:
        dot = sum(l * r for l, r in zip(left, right, strict=False))
        norm_left = sqrt(sum(l * l for l in left))
        norm_right = sqrt(sum(r * r for r in right))
        if norm_left == 0 or norm_right == 0:
            return 0.0
        return dot / (norm_left * norm_right)
    if metric == SimilarityMetric.EUCLIDEAN:
        return sqrt(sum((l - r) ** 2 for l, r in zip(left, right, strict=False)))
    if metric == SimilarityMetric.MANHATTAN:
        return sum(abs(l - r) for l, r in zip(left, right, strict=False))
    return 0.0


def _compare_values(
    left: OrderedPropertyValue | None,
    right: OrderedPropertyValue | None,
) -> int:
    if left is None and right is None:
        return 0
    if left is None:
        return 1
    if right is None:
        return -1
    if left == right:
        return 0
    result = _compare_ordered(left, right)
    if result is None:
        return 0
    return result


def _passes_starting_at(
    node: Node,
    properties: list[str],
    starting_at: list[OrderedPropertyValue | None],
    order_ascending: list[bool],
    include_equal_start: bool,
) -> bool:
    if not starting_at or all(value is None for value in starting_at):
        return True

    for prop, start_value, asc in zip(
        properties, starting_at, order_ascending, strict=False
    ):
        if start_value is None:
            continue
        node_value = _as_ordered_value(node.properties.get(prop))
        if node_value is None:
            return False

        if node_value == start_value:
            continue

        result = _compare_ordered(node_value, start_value)
        if result is None:
            return False
        if asc:
            return result > 0
        return result < 0

    return include_equal_start


def _as_ordered_value(value: PropertyValue) -> OrderedPropertyValue | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, str, datetime)):
        return value
    return None


def _compare_ordered(
    left: OrderedPropertyValue,
    right: OrderedPropertyValue,
) -> int | None:
    if isinstance(left, bool) or isinstance(right, bool):
        return None
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        left_num = float(left)
        right_num = float(right)
        if left_num == right_num:
            return 0
        return -1 if left_num < right_num else 1
    if isinstance(left, str) and isinstance(right, str):
        if left == right:
            return 0
        return -1 if left < right else 1
    if isinstance(left, datetime) and isinstance(right, datetime):
        if left == right:
            return 0
        return -1 if left < right else 1
    return None


def _compare_filter_values(
    left: PropertyValue,
    right: FilterablePropertyValue,
) -> int | None:
    left_value = _as_ordered_value(left)
    right_value = _as_ordered_value(right)
    if left_value is None or right_value is None:
        return None
    return _compare_ordered(left_value, right_value)
