from datetime import UTC, datetime, timedelta
from datetime import UTC, datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine.common.vector_graph_store.sqlite_vector_graph_store import (
    SqliteVectorGraphStore,
    SqliteVectorGraphStoreParams,
)
from memmachine.common.vector_graph_store.data_types import Edge, Node


@pytest_asyncio.fixture
async def sqlite_engine(tmp_path):
    db_path = tmp_path / "vector_graph_store.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def vector_graph_store(sqlite_engine):
    store = SqliteVectorGraphStore(
        SqliteVectorGraphStoreParams(
            engine=sqlite_engine,
        )
    )
    try:
        yield store
    finally:
        await store.close()


@pytest_asyncio.fixture
async def vector_graph_store_ann(vector_graph_store):
    return vector_graph_store


@pytest_asyncio.fixture(autouse=True)
async def db_cleanup(vector_graph_store):
    await vector_graph_store.delete_all_data()
    try:
        yield
    finally:
        await vector_graph_store.delete_all_data()


@pytest.mark.asyncio
async def test_add_nodes(vector_graph_store):
    records = await vector_graph_store.search_matching_nodes(collection="Entity")
    assert len(records) == 0

    nodes = []
    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)

    records = await vector_graph_store.search_matching_nodes(collection="Entity")
    assert len(records) == 0

    nodes = [
        Node(
            uid=str(uuid4()),
            properties={"name": "Node1"},
        ),
        Node(
            uid=str(uuid4()),
            properties={"name": "Node2"},
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Node3",
                "time": datetime.now(tz=UTC),
                "none_value": None,
            },
            embeddings={
                "embedding_name": (
                    [0.1, 0.2, 0.3],
                    SimilarityMetric.COSINE,
                ),
            },
        ),
    ]

    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)

    records = await vector_graph_store.search_matching_nodes(collection="Entity")
    assert len(records) == len(nodes)


@pytest.mark.asyncio
async def test_add_edges(vector_graph_store):
    node1_uid = str(uuid4())
    node2_uid = str(uuid4())
    node3_uid = str(uuid4())

    nodes = [
        Node(
            uid=node1_uid,
            properties={"name": "Node1"},
        ),
        Node(
            uid=node2_uid,
            properties={"name": "Node2"},
        ),
        Node(
            uid=node3_uid,
            properties={
                "name": "Node3",
                "time": datetime.now(tz=UTC),
                "none_value": None,
            },
            embeddings={
                "embedding_name": (
                    [0.1, 0.2, 0.3],
                    SimilarityMetric.COSINE,
                ),
            },
        ),
    ]

    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)

    related_nodes = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node1_uid,
    )
    assert len(related_nodes) == 0

    edges = []
    await vector_graph_store.add_edges(
        relation="RELATED_TO",
        source_collection="Entity",
        target_collection="Entity",
        edges=edges,
    )

    related_nodes = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node1_uid,
    )
    assert len(related_nodes) == 0

    related_to_edges = [
        Edge(
            uid=str(uuid4()),
            source_uid=node1_uid,
            target_uid=node2_uid,
            properties={"description": "Node1 to Node2", "time": datetime.now(tz=UTC)},
        ),
        Edge(
            uid=str(uuid4()),
            source_uid=node2_uid,
            target_uid=node1_uid,
            properties={"description": "Node2 to Node1"},
        ),
        Edge(
            uid=str(uuid4()),
            source_uid=node1_uid,
            target_uid=node3_uid,
            properties={"description": "Node1 to Node3"},
            embeddings={
                "embedding_name": (
                    [0.4, 0.5, 0.6],
                    SimilarityMetric.DOT,
                ),
            },
        ),
    ]

    is_edges = [
        Edge(
            uid=str(uuid4()),
            source_uid=node1_uid,
            target_uid=node1_uid,
            properties={"description": "Node1 loop"},
        ),
        Edge(
            uid=str(uuid4()),
            source_uid=node2_uid,
            target_uid=node2_uid,
            properties={"description": "Node2 loop"},
        ),
    ]

    await vector_graph_store.add_edges(
        relation="RELATED_TO",
        source_collection="Entity",
        target_collection="Entity",
        edges=related_to_edges,
    )
    await vector_graph_store.add_edges(
        relation="IS",
        source_collection="Entity",
        target_collection="Entity",
        edges=is_edges,
    )

    related_nodes = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node1_uid,
    )
    related_names = {node.properties["name"] for node in related_nodes}
    assert {"Node2", "Node3"}.issubset(related_names)


@pytest.mark.asyncio
async def test_search_similar_nodes(vector_graph_store, vector_graph_store_ann):
    nodes = [
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Node1",
            },
            embeddings={
                "embedding1": (
                    [1000.0, 0.0],
                    SimilarityMetric.COSINE,
                ),
                "embedding2": (
                    [1000.0, 0.0],
                    SimilarityMetric.EUCLIDEAN,
                ),
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Node2",
                "include?": "yes",
            },
            embeddings={
                "embedding1": (
                    [10.0, 10.0],
                    SimilarityMetric.COSINE,
                ),
                "embedding2": (
                    [10.0, 10.0],
                    SimilarityMetric.EUCLIDEAN,
                ),
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Node3",
                "include?": "no",
            },
            embeddings={
                "embedding1": (
                    [-100.0, 0.0],
                    SimilarityMetric.COSINE,
                ),
                "embedding2": (
                    [-100.0, 0.0],
                    SimilarityMetric.EUCLIDEAN,
                ),
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Node4",
                "include?": "no",
            },
            embeddings={
                "embedding1": (
                    [-100.0, -1.0],
                    SimilarityMetric.COSINE,
                ),
                "embedding2": (
                    [-100.0, -1.0],
                    SimilarityMetric.EUCLIDEAN,
                ),
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Node5",
                "include?": "no",
            },
            embeddings={
                "embedding1": (
                    [-100.0, -2.0],
                    SimilarityMetric.COSINE,
                ),
                "embedding2": (
                    [-100.0, -2.0],
                    SimilarityMetric.EUCLIDEAN,
                ),
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Node6",
                "include?": "no",
            },
            embeddings={
                "embedding1": (
                    [-100.0, -3.0],
                    SimilarityMetric.COSINE,
                ),
                "embedding2": (
                    [-100.0, -3.0],
                    SimilarityMetric.EUCLIDEAN,
                ),
            },
        ),
    ]

    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)

    results = await vector_graph_store_ann.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=5,
    )
    assert 0 < len(results) <= 5

    results = await vector_graph_store.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=5,
    )
    assert len(results) == 5
    assert results[0].properties["name"] == "Node1"

    results = await vector_graph_store.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=5,
        property_filter=FilterComparison(
            field="include?",
            op="=",
            value="yes",
        ),
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node2"

    results = await vector_graph_store.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=5,
        property_filter=FilterOr(
            left=FilterComparison(
                field="include?",
                op="=",
                value="yes",
            ),
            right=FilterComparison(
                field="include?",
                op="is_null",
                value=None,
            ),
        ),
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Node1"

    results = await vector_graph_store.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding2",
        similarity_metric=SimilarityMetric.EUCLIDEAN,
        limit=5,
    )
    assert len(results) == 5
    assert results[0].properties["name"] == "Node2"

    results = await vector_graph_store.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding2",
        similarity_metric=SimilarityMetric.EUCLIDEAN,
        limit=5,
        property_filter=FilterComparison(
            field="include?",
            op="=",
            value="yes",
        ),
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node2"

    results = await vector_graph_store_ann.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=5,
    )
    assert 0 < len(results) <= 5

    results = await vector_graph_store_ann.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding2",
        similarity_metric=SimilarityMetric.EUCLIDEAN,
        limit=5,
    )
    assert 0 < len(results) <= 5


@pytest.mark.asyncio
async def test_search_related_nodes(vector_graph_store):
    node1_uid = str(uuid4())
    node2_uid = str(uuid4())
    node3_uid = str(uuid4())
    node4_uid = str(uuid4())

    nodes = [
        Node(
            uid=node1_uid,
            properties={"name": "Node1"},
        ),
        Node(
            uid=node2_uid,
            properties={"name": "Node2", "extra!": "something"},
        ),
        Node(
            uid=node3_uid,
            properties={"name": "Node3", "marker?": "A"},
        ),
        Node(
            uid=node4_uid,
            properties={"name": "Node4", "marker?": "B"},
        ),
    ]

    related_to_edges = [
        Edge(
            uid=str(uuid4()),
            source_uid=node1_uid,
            target_uid=node2_uid,
            properties={"description": "Node1 to Node2"},
        ),
        Edge(
            uid=str(uuid4()),
            source_uid=node2_uid,
            target_uid=node1_uid,
            properties={"description": "Node2 to Node1"},
        ),
        Edge(
            uid=str(uuid4()),
            source_uid=node3_uid,
            target_uid=node2_uid,
            properties={
                "description": "Node3 to Node2",
                "extra": 1,
            },
        ),
        Edge(
            uid=str(uuid4()),
            source_uid=node3_uid,
            target_uid=node4_uid,
            properties={
                "description": "Node3 to Node4",
                "extra": 2,
            },
        ),
    ]

    is_edges = [
        Edge(
            uid=str(uuid4()),
            source_uid=node1_uid,
            target_uid=node1_uid,
            properties={"description": "Node1 loop"},
        ),
        Edge(
            uid=str(uuid4()),
            source_uid=node2_uid,
            target_uid=node2_uid,
            properties={"description": "Node2 loop"},
        ),
        Edge(
            uid=str(uuid4()),
            source_uid=node3_uid,
            target_uid=node3_uid,
            properties={"description": "Node3 loop"},
        ),
    ]

    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)
    await vector_graph_store.add_edges(
        relation="RELATED_TO",
        source_collection="Entity",
        target_collection="Entity",
        edges=related_to_edges,
    )
    await vector_graph_store.add_edges(
        relation="RELATED_TO",
        source_collection="Entity",
        target_collection="Entity",
        edges=is_edges,
    )

    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node1_uid,
    )
    assert len(results) == 2
    assert results[0].properties["name"] != results[1].properties["name"]
    assert results[0].properties["name"] in ("Node1", "Node2")
    assert results[1].properties["name"] in ("Node1", "Node2")

    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node1_uid,
        node_property_filter=FilterComparison(
            field="extra!",
            op="=",
            value="something",
        ),
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node2"

    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node2_uid,
        find_sources=False,
    )
    assert len(results) == 2
    assert results[0].properties["name"] != results[1].properties["name"]
    assert results[0].properties["name"] in ("Node1", "Node2")
    assert results[1].properties["name"] in ("Node1", "Node2")

    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node3_uid,
        find_targets=False,
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node3"

    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node3_uid,
        node_property_filter=FilterComparison(
            field="marker?",
            op="=",
            value="A",
        ),
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node3"

    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node3_uid,
        node_property_filter=FilterOr(
            left=FilterComparison(
                field="marker?",
                op="=",
                value="A",
            ),
            right=FilterComparison(
                field="marker?",
                op="is_null",
                value=None,
            ),
        ),
    )
    assert len(results) == 2

    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node3_uid,
        edge_property_filter=FilterComparison(
            field="extra",
            op="=",
            value=1,
        ),
    )
    assert len(results) == 1

    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node3_uid,
        edge_property_filter=FilterOr(
            left=FilterComparison(
                field="extra",
                op="=",
                value=1,
            ),
            right=FilterComparison(
                field="extra",
                op="is_null",
                value=None,
            ),
        ),
    )
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_directional_nodes(vector_graph_store):
    time = datetime.now(tz=UTC)
    delta = timedelta(days=1)

    nodes = [
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event1",
                "timestamp": time,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event2",
                "timestamp": time + delta,
                "include?": "yes",
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event3",
                "timestamp": time + 2 * delta,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event4",
                "timestamp": time + 3 * delta,
                "include?": "yes",
            },
        ),
    ]

    await vector_graph_store.add_nodes(collection="Event", nodes=nodes)

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[time + delta],
        order_ascending=[True],
        include_equal_start=True,
        limit=2,
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event2"
    assert results[1].properties["name"] == "Event3"

    result_timestamp = (
        results[0].properties["timestamp"].astimezone(ZoneInfo("America/Los_Angeles"))
    )

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[result_timestamp],
        order_ascending=[False],
        include_equal_start=False,
        limit=1,
    )
    assert len(results) == 1
    assert results[0].properties["name"] != "Event2"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[result_timestamp],
        order_ascending=[True],
        include_equal_start=False,
        limit=1,
    )
    assert len(results) == 1
    assert results[0].properties["name"] != "Event2"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[time + delta],
        order_ascending=[True],
        include_equal_start=True,
        limit=2,
        property_filter=FilterComparison(
            field="include?",
            op="=",
            value="yes",
        ),
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event2"
    assert results[1].properties["name"] == "Event4"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[time + delta],
        order_ascending=[False],
        include_equal_start=True,
        limit=2,
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event2"
    assert results[1].properties["name"] == "Event1"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[time + delta],
        order_ascending=[True],
        include_equal_start=False,
        limit=2,
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event3"
    assert results[1].properties["name"] == "Event4"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[time + delta],
        order_ascending=[False],
        include_equal_start=False,
        limit=2,
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Event1"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[None],
        order_ascending=[False],
        limit=2,
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event4"
    assert results[1].properties["name"] == "Event3"


@pytest.mark.asyncio
async def test_search_directional_nodes_multiple_by_properties(vector_graph_store):
    time = datetime.now(tz=UTC)
    delta = timedelta(days=1)

    nodes = [
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event1",
                "timestamp": time,
                "pair": 1,
                "sequence": 1,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event2",
                "timestamp": time,
                "pair": 1,
                "sequence": 2,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event3",
                "timestamp": time,
                "pair": 2,
                "sequence": 1,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event4",
                "timestamp": time,
                "pair": 2,
                "sequence": 2,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event5",
                "timestamp": time,
                "pair": 3,
                "sequence": 1,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event6",
                "timestamp": time,
                "pair": 3,
                "sequence": 2,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event7",
                "timestamp": time + delta,
                "pair": 1,
                "sequence": 1,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event8",
                "timestamp": time + delta,
                "pair": 1,
                "sequence": 2,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event9",
                "timestamp": time + delta,
                "pair": 2,
                "sequence": 1,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event10",
                "timestamp": time + delta,
                "pair": 2,
                "sequence": 2,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event11",
                "timestamp": time + delta,
                "pair": 3,
                "sequence": 1,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event12",
                "timestamp": time + delta,
                "pair": 3,
                "sequence": 2,
            },
        ),
    ]

    await vector_graph_store.add_nodes(collection="Event", nodes=nodes)
    records = await vector_graph_store.search_matching_nodes(collection="Event")
    assert len(records) == 12

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp", "pair", "sequence"],
        starting_at=[time + delta, 1, 2],
        order_ascending=[True, True, True],
        include_equal_start=True,
        limit=None,
    )
    assert len(results) == 5
    assert results[0].properties["name"] == "Event8"
    assert results[1].properties["name"] == "Event9"
    assert results[2].properties["name"] == "Event10"
    assert results[3].properties["name"] == "Event11"
    assert results[4].properties["name"] == "Event12"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp", "pair", "sequence"],
        starting_at=[time + delta, 1, 2],
        order_ascending=[True, True, False],
        include_equal_start=True,
        limit=None,
    )
    assert len(results) == 6
    assert results[0].properties["name"] == "Event8"
    assert results[1].properties["name"] == "Event7"
    assert results[2].properties["name"] == "Event10"
    assert results[3].properties["name"] == "Event9"
    assert results[4].properties["name"] == "Event12"
    assert results[5].properties["name"] == "Event11"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp", "pair", "sequence"],
        starting_at=[time + delta, 1, 2],
        order_ascending=[True, False, True],
        include_equal_start=True,
        limit=None,
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Event8"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp", "pair", "sequence"],
        starting_at=[time + delta, 1, 2],
        order_ascending=[True, False, False],
        include_equal_start=True,
        limit=None,
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event8"
    assert results[1].properties["name"] == "Event7"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp", "pair", "sequence"],
        starting_at=[time + delta, 1, 2],
        order_ascending=[False, True, True],
        include_equal_start=True,
        limit=None,
    )
    assert len(results) == 11
    assert results[0].properties["name"] == "Event8"
    assert results[1].properties["name"] == "Event9"
    assert results[2].properties["name"] == "Event10"
    assert results[3].properties["name"] == "Event11"
    assert results[4].properties["name"] == "Event12"
    assert results[5].properties["name"] == "Event1"
    assert results[6].properties["name"] == "Event2"
    assert results[7].properties["name"] == "Event3"
    assert results[8].properties["name"] == "Event4"
    assert results[9].properties["name"] == "Event5"
    assert results[10].properties["name"] == "Event6"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp", "pair", "sequence"],
        starting_at=[time + delta, 1, 2],
        order_ascending=[False, True, False],
        include_equal_start=True,
        limit=None,
    )
    assert len(results) == 12
    assert results[0].properties["name"] == "Event8"
    assert results[1].properties["name"] == "Event7"
    assert results[2].properties["name"] == "Event10"
    assert results[3].properties["name"] == "Event9"
    assert results[4].properties["name"] == "Event12"
    assert results[5].properties["name"] == "Event11"
    assert results[6].properties["name"] == "Event2"
    assert results[7].properties["name"] == "Event1"
    assert results[8].properties["name"] == "Event4"
    assert results[9].properties["name"] == "Event3"
    assert results[10].properties["name"] == "Event6"
    assert results[11].properties["name"] == "Event5"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp", "pair", "sequence"],
        starting_at=[time + delta, 1, 2],
        order_ascending=[False, False, True],
        include_equal_start=True,
        limit=None,
    )
    assert len(results) == 7
    assert results[0].properties["name"] == "Event8"
    assert results[1].properties["name"] == "Event5"
    assert results[2].properties["name"] == "Event6"
    assert results[3].properties["name"] == "Event3"
    assert results[4].properties["name"] == "Event4"
    assert results[5].properties["name"] == "Event1"
    assert results[6].properties["name"] == "Event2"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp", "pair", "sequence"],
        starting_at=[time + delta, 1, 2],
        order_ascending=[False, False, False],
        include_equal_start=True,
        limit=None,
    )
    assert len(results) == 8
    assert results[0].properties["name"] == "Event8"
    assert results[1].properties["name"] == "Event7"
    assert results[2].properties["name"] == "Event6"
    assert results[3].properties["name"] == "Event5"
    assert results[4].properties["name"] == "Event4"
    assert results[5].properties["name"] == "Event3"
    assert results[6].properties["name"] == "Event2"
    assert results[7].properties["name"] == "Event1"


@pytest.mark.asyncio
async def test_search_matching_nodes(vector_graph_store):
    person_nodes = [
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Alice",
                "age!with$pecialchars": 30,
                "city": "San Francisco",
                "title": "Engineer",
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Bob",
                "age!with$pecialchars": 25,
                "city": "Los Angeles",
                "title": "Designer",
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Charlie",
                "city": "New York",
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "David",
                "age!with$pecialchars": 30,
                "city": "New York",
                "none_value": None,
            },
        ),
    ]

    robot_nodes = [
        Node(
            uid=str(uuid4()),
            properties={"name": "Eve", "city": "Axiom"},
        ),
    ]

    await vector_graph_store.add_nodes(collection="Person", nodes=person_nodes)
    await vector_graph_store.add_nodes(collection="Robot", nodes=robot_nodes)

    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
    )
    assert len(results) == 4

    results = await vector_graph_store.search_matching_nodes(
        collection="Robot",
    )
    assert len(results) == 1

    results = await vector_graph_store.search_matching_nodes(
        collection="Robot",
        property_filter=FilterComparison(
            field="none_value",
            op="is_null",
            value=None,
        ),
    )
    assert len(results) == 1

    results = await vector_graph_store.search_matching_nodes(
        collection="Robot",
        property_filter=FilterComparison(
            field="none_value",
            op="=",
            value="something",
        ),
    )
    assert len(results) == 0

    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        property_filter=FilterComparison(
            field="city",
            op="=",
            value="New York",
        ),
    )
    assert len(results) == 2

    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        property_filter=FilterAnd(
            left=FilterComparison(
                field="city",
                op="=",
                value="San Francisco",
            ),
            right=FilterComparison(
                field="age!with$pecialchars",
                op="=",
                value=20,
            ),
        ),
    )
    assert len(results) == 0

    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        property_filter=FilterAnd(
            left=FilterComparison(
                field="city",
                op="=",
                value="New York",
            ),
            right=FilterComparison(
                field="age!with$pecialchars",
                op="=",
                value=30,
            ),
        ),
    )
    assert len(results) == 1

    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        property_filter=FilterComparison(
            field="age!with$pecialchars",
            op="=",
            value=30,
        ),
    )
    assert len(results) == 2

    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        property_filter=FilterOr(
            left=FilterComparison(
                field="age!with$pecialchars",
                op="=",
                value=30,
            ),
            right=FilterComparison(
                field="age!with$pecialchars",
                op="is_null",
                value=None,
            ),
        ),
    )
    assert len(results) == 3

    # Should only include Alice.
    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        property_filter=FilterComparison(
            field="title",
            op="=",
            value="Engineer",
        ),
    )
    assert len(results) == 1

    # Should include Alice and all Person nodes without the "title" property.
    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        property_filter=FilterOr(
            left=FilterComparison(
                field="title",
                op="=",
                value="Engineer",
            ),
            right=FilterComparison(
                field="title",
                op="is_null",
                value=None,
            ),
        ),
    )
    assert len(results) == 3


@pytest.mark.asyncio
async def test_get_nodes(vector_graph_store):
    nodes = [
        Node(
            uid=str(uuid4()),
            properties={"name": "Node1", "time": datetime.now(tz=UTC)},
        ),
        Node(
            uid=str(uuid4()),
            properties={"name": "Node2"},
        ),
        Node(
            uid=str(uuid4()),
            properties={"name": "Node3"},
        ),
    ]

    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)

    fetched_nodes = await vector_graph_store.get_nodes(
        collection="Entity",
        node_uids=[node.uid for node in nodes],
    )
    assert len(fetched_nodes) == 3

    for fetched_node in fetched_nodes:
        assert fetched_node.uid in {node.uid for node in nodes}

    fetched_nodes = await vector_graph_store.get_nodes(
        collection="Entity",
        node_uids=[nodes[0].uid, uuid4()],
    )
    assert len(fetched_nodes) == 1
    assert fetched_nodes[0] == nodes[0]


@pytest.mark.asyncio
async def test_delete_nodes(vector_graph_store):
    nodes = [
        Node(
            uid=str(uuid4()),
        ),
        Node(
            uid=str(uuid4()),
        ),
        Node(
            uid=str(uuid4()),
        ),
        Node(
            uid=str(uuid4()),
        ),
        Node(
            uid=str(uuid4()),
        ),
        Node(
            uid=str(uuid4()),
        ),
    ]

    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)
    records = await vector_graph_store.search_matching_nodes(collection="Entity")
    assert len(records) == 6

    await vector_graph_store.delete_nodes(
        collection="Bad", node_uids=[node.uid for node in nodes[:-3]]
    )
    records = await vector_graph_store.search_matching_nodes(collection="Entity")
    assert len(records) == 6

    await vector_graph_store.delete_nodes(
        collection="Entity", node_uids=[node.uid for node in nodes[:-3]]
    )
    records = await vector_graph_store.search_matching_nodes(collection="Entity")
    assert len(records) == 3


@pytest.mark.asyncio
async def test_delete_all_data(vector_graph_store):
    nodes = [
        Node(
            uid=str(uuid4()),
        ),
        Node(
            uid=str(uuid4()),
        ),
        Node(
            uid=str(uuid4()),
        ),
        Node(
            uid=str(uuid4()),
        ),
        Node(
            uid=str(uuid4()),
        ),
        Node(
            uid=str(uuid4()),
        ),
    ]

    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)
    records = await vector_graph_store.search_matching_nodes(collection="Entity")
    assert len(records) == 6

    await vector_graph_store.delete_all_data()
    records = await vector_graph_store.search_matching_nodes(collection="Entity")
    assert len(records) == 0
