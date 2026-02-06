# MemMachine Rust

High-performance Rust implementation of the MemMachine memory layer for AI agents.

## Building

### Prerequisites

- Rust 1.75+ (install via [rustup](https://rustup.rs/))
- Python 3.10+ (for Python bindings)
- maturin (`pip install maturin`)

### Development Build

```bash
# Build and install in development mode
cd src/memmachine-rs
maturin develop

# Or with release optimizations
maturin develop --release
```

### Production Build

```bash
maturin build --release
pip install target/wheels/memmachine_rust-*.whl
```

## Testing

### Rust Tests

```bash
cargo test
```

### Python Tests

```bash
# After building with maturin
cd tests/python
pytest
```

## Usage

### Python

```python
import asyncio
from memmachine_rust import (
    MemMachine,
    SessionData,
    EpisodeEntry,
    EpisodeRole,
    MemoryType,
)

async def main():
    # Create MemMachine instance
    mm = MemMachine("/path/to/storage.db")
    
    # Start the service
    await mm.start()
    
    # Create a session
    info = await mm.create_session("my-session", "My chat session")
    
    # Add episodes
    session_data = SessionData("my-session")
    entries = [
        EpisodeEntry(EpisodeRole.User, "Hello!"),
        EpisodeEntry(EpisodeRole.Assistant, "Hi there!"),
    ]
    
    episode_ids = await mm.add_episodes(
        session_data,
        entries,
        [MemoryType.Episodic]
    )
    
    # Search
    results = await mm.query_search(
        session_data,
        "greeting",
        [MemoryType.Episodic],
        limit=10
    )
    
    # Stop when done
    await mm.stop()

asyncio.run(main())
```

### Rust

```rust
use memmachine_core::{
    Configuration, MemMachine, SessionData, EpisodeEntry, EpisodeRole, MemoryType
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create configuration
    let config = Configuration::new("/path/to/storage.db");
    
    // Create MemMachine instance
    let mut mm = MemMachine::new(config).await?;
    
    // Start the service
    mm.start().await?;
    
    // Create a session
    let info = mm.create_session("my-session", "My chat session").await?;
    
    // Add episodes
    let session_data = SessionData::new("my-session");
    let entries = vec![
        EpisodeEntry::new(EpisodeRole::User, "Hello!"),
        EpisodeEntry::new(EpisodeRole::Assistant, "Hi there!"),
    ];
    
    let episode_ids = mm.add_episodes(
        &session_data,
        entries,
        &[MemoryType::Episodic]
    ).await?;
    
    // Stop when done
    mm.stop().await?;
    
    Ok(())
}
```

## Architecture

```
memmachine-rs/
├── Cargo.toml                    # Workspace root
├── pyproject.toml                # Python build config
├── crates/
│   ├── memmachine-core/          # Core Rust library
│   │   └── src/
│   │       ├── lib.rs            # Public exports
│   │       ├── config.rs         # Configuration types
│   │       ├── error.rs          # Error types
│   │       ├── memmachine.rs     # Main orchestrator
│   │       ├── types/            # Core data types
│   │       ├── storage/          # SQLite storage layer
│   │       ├── embedder/         # Embedding providers
│   │       └── llm/              # LLM clients
│   └── memmachine-py/            # PyO3 Python bindings
│       └── src/
│           ├── lib.rs            # Python module
│           ├── types.rs          # Type conversions
│           └── memmachine.rs     # PyMemMachine wrapper
├── python/
│   └── memmachine_rust/          # Python package
│       ├── __init__.py
│       └── memmachine_rust.pyi   # Type stubs
└── tests/
    ├── integration_tests.rs      # Rust integration tests
    └── python/
        └── test_memmachine.py    # Python integration tests
```

## Configuration

Configuration can be provided via YAML file:

```yaml
storage:
  path: /path/to/memmachine.db
  wal_mode: true
  max_connections: 5

episodic_memory:
  enabled: true
  long_term_memory:
    enabled: true
    embedder: openai-embedder

semantic_memory:
  enabled: true
  embedder: openai-embedder

embedders:
  - provider: openai
    name: openai-embedder
    api_key: ${OPENAI_API_KEY}
    model: text-embedding-3-small

language_models:
  - provider: openai
    name: gpt-4
    api_key: ${OPENAI_API_KEY}
    model: gpt-4-turbo-preview
```

## License

Apache 2.0
