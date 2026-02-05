"""Utility functions and constants for MemMachine installation scripts."""

from enum import Enum


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_BEDROCK_MODEL = "openai.gpt-oss-20b-1:0"
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_BEDROCK_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_EMBEDDING_DIMENSIONS = 768

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class ModelProvider(Enum):
    """Enumeration of supported language model providers."""

    OPENAI = "openai"
    BEDROCK = "bedrock"
    OLLAMA = "ollama"

    @classmethod
    def parse(cls, raw: str) -> "ModelProvider":
        """
        Parse user-provided input (case-insensitive) and map it to a ModelProvider.

        Falls back to OPENAI on invalid input.
        """
        if not raw:
            return cls.OPENAI

        raw = raw.strip().lower()

        mapping = {
            "openai": cls.OPENAI,
            "bedrock": cls.BEDROCK,
            "ollama": cls.OLLAMA,
        }

        return mapping.get(raw, cls.OPENAI)
