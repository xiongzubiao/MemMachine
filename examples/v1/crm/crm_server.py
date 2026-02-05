import logging
import os
from datetime import UTC, datetime

import aiosqlite
import requests
from dotenv import load_dotenv
from fastapi import FastAPI

try:
    from .query_constructor import CRMQueryConstructor
except ImportError:
    from examples.v1.crm.query_constructor import CRMQueryConstructor

logger = logging.getLogger(__name__)

load_dotenv()

MEMORY_BACKEND_URL = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
CRM_PORT = int(os.getenv("CRM_PORT", "8000"))

CRM_DEDUPE_DB = os.getenv("CRM_DEDUPE_DB", "crm_dedupe.db")

app = FastAPI(title="Server", description="Simple middleware")

query_constructor = CRMQueryConstructor()


async def _ensure_dedupe_table(conn: aiosqlite.Connection) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS slack_dedupe (
            slack_message_id TEXT PRIMARY KEY
        )
        """
    )
    await conn.commit()


async def mark_slack_message_processed(slack_message_id: str) -> None:
    async with aiosqlite.connect(CRM_DEDUPE_DB) as conn:
        await _ensure_dedupe_table(conn)
        await conn.execute(
            "INSERT OR IGNORE INTO slack_dedupe (slack_message_id) VALUES (?)",
            (slack_message_id,),
        )
        await conn.commit()


async def is_slack_message_processed(
    slack_message_id: str,
    user_id: str,
    session_id: str,
) -> bool:
    """Check if Slack message was already processed by querying history table metadata"""
    try:
        async with aiosqlite.connect(CRM_DEDUPE_DB) as conn:
            await _ensure_dedupe_table(conn)
            async with conn.execute(
                "SELECT 1 FROM slack_dedupe WHERE slack_message_id = ?",
                (slack_message_id,),
            ) as cursor:
                result = await cursor.fetchone()
            if result:
                print(
                    "[CRM] Found duplicate slack_message_id in dedupe store: "
                    f"{slack_message_id}",
                )
            return result is not None
    except Exception:
        logger.exception("Error occurred in is_slack_message_processed")
        return False


@app.post("/memory")
async def store_data(user_id: str, query: str, slack_message_id: str | None):
    try:
        if slack_message_id and await is_slack_message_processed(
            slack_message_id,
            user_id,
            f"session_{user_id}",
        ):
            print(f"[CRM] Slack message {slack_message_id} already processed, skipping")
            return {"status": "skipped", "message": "Message already processed"}

        session_data = {
            "group_id": user_id,
            "agent_id": ["assistant"],
            "user_id": [user_id],
            "session_id": f"session_{user_id}",
        }
        episode_data = {
            "session": session_data,
            "producer": user_id,
            "produced_for": "assistant",
            "episode_content": query,
            "episode_type": "message",
            "metadata": {
                "speaker": user_id,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "type": "message",
                "slack_message_id": slack_message_id,
            },
        }

        response = requests.post(
            f"{MEMORY_BACKEND_URL}/v1/memories",
            json=episode_data,
            timeout=1000,
        )
        response.raise_for_status()
        if slack_message_id:
            await mark_slack_message_processed(slack_message_id)
        return {"status": "success", "data": response.json()}
    except Exception:
        logger.exception("Error occurred in /memory store_data")
        return {"status": "error", "message": "Internal error in /memory store_data"}


@app.get("/memory")
async def get_data(query: str, user_id: str, timestamp: str):
    try:
        session_data = {
            "group_id": user_id,
            "agent_id": ["assistant"],
            "user_id": [user_id],
            "session_id": f"session_{user_id}",
        }
        search_data = {
            "session": session_data,
            "query": query,
            "limit": 20,
            "filter": {"producer_id": user_id},
        }

        logger.debug(
            "Sending POST request to %s/v1/memories/search",
            MEMORY_BACKEND_URL,
        )
        logger.debug("Search data: %s", search_data)

        response = requests.post(
            f"{MEMORY_BACKEND_URL}/v1/memories/search",
            json=search_data,
            timeout=1000,
        )

        logger.debug("Response status: %s", response.status_code)
        logger.debug("Response headers: %s", dict(response.headers))

        if response.status_code != 200:
            logger.error(
                "Backend returned %s: %s",
                response.status_code,
                response.text,
            )
            return {
                "status": "error",
                "message": "Failed to retrieve memory data",
            }

        response_data = response.json()
        logger.debug("Response data: %s", response_data)

        content = response_data.get("content", {})
        episodic_memory = content.get("episodic_memory", [])
        profile_memory = content.get("profile_memory", [])

        profile_str = ""
        if profile_memory:
            if isinstance(profile_memory, list):
                profile_str = "\n".join([str(p) for p in profile_memory])
            else:
                profile_str = str(profile_memory)

        context_str = ""
        if episodic_memory:
            if isinstance(episodic_memory, list):
                context_str = "\n".join([str(c) for c in episodic_memory])
            else:
                context_str = str(episodic_memory)

        formatted_query = query_constructor.create_query(
            profile=profile_str,
            context=context_str,
            query=query,
        )

        return {
            "status": "success",
            "data": {"profile": profile_memory, "context": episodic_memory},
            "formatted_query": formatted_query,
            "query_type": "example",
        }
    except Exception:
        logger.exception("Error occurred in /memory get_data")
        return {"status": "error", "message": "Internal error in /memory get_data"}


@app.post("/memory/store-and-search")
async def store_and_search_data(user_id: str, query: str):
    try:
        session_data = {
            "group_id": user_id,
            "agent_id": ["assistant"],
            "user_id": [user_id],
            "session_id": f"session_{user_id}",
        }
        episode_data = {
            "session": session_data,
            "producer": user_id,
            "produced_for": "assistant",
            "episode_content": query,
            "episode_type": "message",
            "metadata": {
                "speaker": user_id,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "type": "message",
            },
        }

        resp = requests.post(
            f"{MEMORY_BACKEND_URL}/v1/memories",
            json=episode_data,
            timeout=1000,
        )

        logger.debug("Store-and-search response status: %s", resp.status_code)
        if resp.status_code != 200:
            logger.error("Store failed with %s: %s", resp.status_code, resp.text)
            return {
                "status": "error",
                "message": "Failed to store memory data",
            }

        search_data = {
            "session": session_data,
            "query": query,
            "limit": 5,
            "filter": {"producer_id": user_id},
        }

        search_resp = requests.post(
            f"{MEMORY_BACKEND_URL}/v1/memories/search",
            json=search_data,
            timeout=1000,
        )

        logger.debug(
            "Store-and-search response status: %s",
            search_resp.status_code,
        )
        if search_resp.status_code != 200:
            logger.error(
                "Search failed with %s: %s", search_resp.status_code, search_resp.text
            )
            return {
                "status": "error",
                "message": "Failed to search memory data",
            }

        search_resp.raise_for_status()

        search_results = search_resp.json()

        content = search_results.get("content", {})
        episodic_memory = content.get("episodic_memory", [])
        profile_memory = content.get("profile_memory", [])

        profile_str = ""
        if profile_memory:
            if isinstance(profile_memory, list):
                profile_str = "\n".join([str(p) for p in profile_memory])
            else:
                profile_str = str(profile_memory)

        context_str = ""
        if episodic_memory:
            if isinstance(episodic_memory, list):
                context_str = "\n".join([str(c) for c in episodic_memory])
            else:
                context_str = str(episodic_memory)

        formatted_response = query_constructor.create_query(
            profile=profile_str,
            context=context_str,
            query=query,
        )

        if profile_memory and episodic_memory:
            return f"Profile: {profile_memory}\n\nContext: {episodic_memory}\n\nFormatted Response:\n{formatted_response}"
        if profile_memory:
            return f"Profile: {profile_memory}\n\nFormatted Response:\n{formatted_response}"
        if episodic_memory:
            return f"Context: {episodic_memory}\n\nFormatted Response:\n{formatted_response}"
        return f"Message ingested successfully. No relevant context found yet.\n\nFormatted Response:\n{formatted_response}"

    except Exception:
        logger.exception("Error occurred in /memory store-and-search")
        return {"status": "error", "message": "Internal error in store_and_search"}


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the dedupe store on startup."""
    async with aiosqlite.connect(CRM_DEDUPE_DB) as conn:
        await _ensure_dedupe_table(conn)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=CRM_PORT)
