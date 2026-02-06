from typing import Any, Awaitable

from .memmachine_rust import (
    EpisodeRole as EpisodeRole,
    MemoryType as MemoryType,
    EpisodeEntry as EpisodeEntry,
    Episode as Episode,
    SessionData as SessionData,
    SessionInfo as SessionInfo,
    Session as Session,
    SemanticFeature as SemanticFeature,
    FeatureMetadata as FeatureMetadata,
    EpisodicSearchResult as EpisodicSearchResult,
    SearchResponse as SearchResponse,
    ListResults as ListResults,
    MemMachine as MemMachine,
)

__version__: str
__all__ = [
    "__version__",
    "EpisodeRole",
    "MemoryType",
    "EpisodeEntry",
    "Episode",
    "SessionData",
    "SessionInfo",
    "Session",
    "SemanticFeature",
    "FeatureMetadata",
    "EpisodicSearchResult",
    "SearchResponse",
    "ListResults",
    "MemMachine",
]
