"""
Memory Service for Shvayambhu
==============================

Simple Python wrapper for memory storage functionality.
In production, this would interface with the NestJS/GraphQL API.
"""

import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class MemoryService:
    """Memory storage and retrieval service."""
    
    def __init__(self, db_path: str = "data/shvayambhu.db"):
        """Initialize memory service with SQLite database."""
        self.db_path = db_path
        self._ensure_database()
        
    def _ensure_database(self):
        """Ensure database and tables exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp 
            ON memories(timestamp DESC)
        """)
        conn.commit()
        conn.close()
        
    async def store_memory(self, memory: Dict[str, Any]) -> int:
        """Store a memory in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO memories (content, embedding, metadata)
            VALUES (?, ?, ?)
        """, (
            json.dumps(memory, cls=DateTimeEncoder),
            None,  # Embeddings would be computed here
            json.dumps({
                "prompt": memory.get("prompt", ""),
                "response": memory.get("response", ""),
                "consciousness_state": memory.get("consciousness_state", {}),
                "emotional_state": memory.get("emotional_state", {})
            }, cls=DateTimeEncoder)
        ))
        
        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return memory_id
        
    async def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant memories."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Simple text search - in production would use embeddings
        cursor.execute("""
            SELECT content, metadata, timestamp
            FROM memories
            WHERE content LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"%{query}%", limit))
        
        memories = []
        for row in cursor.fetchall():
            try:
                content = json.loads(row["content"])
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                memories.append({
                    "content": content.get("response", content.get("prompt", "")),
                    "metadata": metadata,
                    "timestamp": row["timestamp"]
                })
            except json.JSONDecodeError:
                continue
                
        conn.close()
        return memories
        
    async def get_recent_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent memories."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT content, metadata, timestamp
            FROM memories
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        memories = []
        for row in cursor.fetchall():
            try:
                content = json.loads(row["content"])
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                memories.append({
                    "content": content,
                    "metadata": metadata,
                    "timestamp": row["timestamp"]
                })
            except json.JSONDecodeError:
                continue
                
        conn.close()
        return memories