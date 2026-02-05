# MemMachine Docker Setup Guide

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- OpenAI API key configured

### 1. Configure Environment
Copy the example environment file and add your OpenAI API key:
```bash
cp sample_configs/env.dockercompose .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Configure MemMachine
Copy the sample configuration file and update it with your settings:
```bash
cp sample_configs/episodic_memory_config.sample configuration.yml
# Edit configuration.yml and update:
# - Replace <YOUR_API_KEY> with your OpenAI API key
```

### 3. Start Services

#### Option A: Using the MemMachine Compose Script (Recommended)
Run the startup script:
```bash
./memmachine-compose.sh
```

This will:
- ✅ Check Docker and Docker Compose availability
- ✅ Verify .env file and OpenAI API key
- ✅ Check and create configuration.yml if needed
- ✅ Validate configuration settings
- ✅ Pull and start services (MemMachine)
- ✅ Wait for all services to be healthy
- ✅ Display service URLs and connection info

#### Option B: Using Docker Compose Directly
```bash
docker-compose up -d
```

### 4. Access Services
Once started, you can access:

- **MemMachine API**: http://localhost:8080
- **Health Check**: http://localhost:8080/health
- **Metrics**: http://localhost:8080/metrics

### 5. Test the Setup
```bash
# Test health endpoint
curl http://localhost:8080/health

# Test memory storage
curl -X POST "http://localhost:8080/v1/memories" \
  -H "Content-Type: application/json" \
  -d '{
    "session": {
      "group_id": "test-group",
      "agent_id": ["test-agent"],
      "user_id": ["test-user"],
      "session_id": "test-session-123"
    },
    "producer": "test-user",
    "produced_for": "test-user",
    "episode_content": "Hello, this is a test message",
    "episode_type": "text",
    "metadata": {"test": true}
  }'
```

## Useful Commands

### Using the MemMachine Compose Script (Recommended)

#### View Logs
```bash
./memmachine-compose.sh logs
```

#### Stop Services
```bash
./memmachine-compose.sh stop
```

#### Restart Services
```bash
./memmachine-compose.sh restart
```

#### Clean Up (Remove All Data)
```bash
./memmachine-compose.sh clean
```

#### Show Help
```bash
./memmachine-compose.sh help
```

### Using Docker Compose Directly

#### View Logs
```bash
docker-compose logs -f
```

#### Stop Services
```bash
docker-compose down
```

#### Restart Services
```bash
docker-compose restart
```

#### Clean Up (Remove All Data)
```bash
docker-compose down -v
```

## Services

- **SQLite profile store**: Local file storage persisted via Docker volume
- **MemMachine** (port 8080): Main API server (uses pre-built `memmachine/memmachine` image)

## Configuration

Key files:
- `.env` - Environment variables
- `configuration.yml` - MemMachine configuration
- `docker-compose.yml` - Service definitions
- `memmachine-compose.sh` - Startup script with validation and health checks

### ⚠️ Important Configuration Notes

**1. Database Configuration Consistency**
Make sure the database configuration details in `configuration.yml` match the database configuration details in `.env`

Both files must have consistent:
- SQLite file paths for profile and graph stores

**2. Configuration.yml Setup**
The `configuration.yml` file contains MemMachine-specific settings:
- **Model configuration**: OpenAI API settings for LLM and embeddings
- **Storage configuration**: SQLite and Chroma settings
- **Memory settings**: Session memory capacity and limits
- **Reranker configuration**: Search and ranking algorithms

**Key settings to update in configuration.yml:**
- Replace `<YOUR_API_KEY>` with your OpenAI API key (appears in both Model and embedder sections)

This ensures MemMachine can properly connect to the Docker services and use your OpenAI API key for embeddings and LLM operations.
