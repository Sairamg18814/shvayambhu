-- Shvayambhu SQLite Database Schema
-- Optimized for compressed storage and consciousness tracking

-- Enable compression and optimize for performance
PRAGMA page_size = 65536;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

-- Consciousness states table
CREATE TABLE IF NOT EXISTS consciousness_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    self_awareness_level REAL CHECK(self_awareness_level >= 0 AND self_awareness_level <= 1),
    emotional_state TEXT, -- JSON compressed
    attention_focus TEXT, -- JSON compressed
    introspective_depth INTEGER CHECK(introspective_depth >= 0),
    phenomenal_model_snapshot BLOB, -- Compressed binary
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Compressed memories table
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL CHECK(type IN ('episodic', 'semantic', 'procedural', 'conscious_experience')),
    original_size INTEGER NOT NULL,
    compressed_size INTEGER NOT NULL,
    compression_ratio REAL GENERATED ALWAYS AS (CAST(original_size AS REAL) / compressed_size) STORED,
    data BLOB NOT NULL, -- LZ4 compressed
    metadata TEXT, -- JSON
    consciousness_state_id INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    last_accessed DATETIME,
    FOREIGN KEY (consciousness_state_id) REFERENCES consciousness_states(id)
);

-- Knowledge graph nodes
CREATE TABLE IF NOT EXISTS knowledge_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    attributes TEXT, -- Compressed JSON
    embeddings BLOB, -- Compressed vector
    source_url TEXT,
    confidence_score REAL CHECK(confidence_score >= 0 AND confidence_score <= 1),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_type, entity_name)
);

-- Knowledge graph edges
CREATE TABLE IF NOT EXISTS knowledge_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_node_id INTEGER NOT NULL,
    target_node_id INTEGER NOT NULL,
    relationship_type TEXT NOT NULL,
    strength REAL DEFAULT 1.0 CHECK(strength >= 0 AND strength <= 1),
    metadata TEXT, -- Compressed JSON
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_node_id) REFERENCES knowledge_nodes(id),
    FOREIGN KEY (target_node_id) REFERENCES knowledge_nodes(id),
    UNIQUE(source_node_id, target_node_id, relationship_type)
);

-- Training experiences table
CREATE TABLE IF NOT EXISTS training_experiences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experience_type TEXT NOT NULL,
    input_data BLOB, -- Compressed
    output_data BLOB, -- Compressed
    loss_value REAL,
    reward_signal REAL,
    model_state_hash TEXT,
    consciousness_insights TEXT, -- Compressed JSON
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Model checkpoints table
CREATE TABLE IF NOT EXISTS model_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_name TEXT UNIQUE NOT NULL,
    model_type TEXT NOT NULL,
    model_size_bytes INTEGER NOT NULL,
    compressed_size_bytes INTEGER NOT NULL,
    compression_ratio REAL GENERATED ALWAYS AS (CAST(model_size_bytes AS REAL) / compressed_size_bytes) STORED,
    weights_data BLOB, -- Compressed model weights
    metadata TEXT, -- JSON with training info
    performance_metrics TEXT, -- JSON
    consciousness_level REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Web intelligence cache
CREATE TABLE IF NOT EXISTS web_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    content_hash TEXT NOT NULL,
    compressed_content BLOB NOT NULL,
    original_size INTEGER NOT NULL,
    compressed_size INTEGER NOT NULL,
    content_type TEXT,
    last_fetched DATETIME DEFAULT CURRENT_TIMESTAMP,
    expiry_time DATETIME,
    access_count INTEGER DEFAULT 0
);

-- Conversation logs with compression
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    user_input TEXT,
    model_response TEXT,
    compressed_context BLOB, -- Compressed conversation context
    consciousness_state_id INTEGER,
    thinking_process TEXT, -- Compressed JSON of internal reasoning
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (consciousness_state_id) REFERENCES consciousness_states(id),
    UNIQUE(session_id, turn_number)
);

-- Compression statistics table
CREATE TABLE IF NOT EXISTS compression_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    total_original_size INTEGER DEFAULT 0,
    total_compressed_size INTEGER DEFAULT 0,
    average_ratio REAL DEFAULT 0,
    best_ratio REAL DEFAULT 0,
    worst_ratio REAL DEFAULT 0,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_entity ON knowledge_nodes(entity_type, entity_name);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_source ON knowledge_edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_target ON knowledge_edges(target_node_id);
CREATE INDEX IF NOT EXISTS idx_consciousness_timestamp ON consciousness_states(timestamp);
CREATE INDEX IF NOT EXISTS idx_web_cache_url ON web_cache(url);
CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);

-- Create triggers for auto-updating timestamps
CREATE TRIGGER IF NOT EXISTS update_knowledge_nodes_timestamp
AFTER UPDATE ON knowledge_nodes
BEGIN
    UPDATE knowledge_nodes SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Create trigger for compression statistics
CREATE TRIGGER IF NOT EXISTS update_compression_stats_on_memory_insert
AFTER INSERT ON memories
BEGIN
    INSERT OR REPLACE INTO compression_stats (table_name, total_original_size, total_compressed_size)
    VALUES (
        'memories',
        COALESCE((SELECT total_original_size FROM compression_stats WHERE table_name = 'memories'), 0) + NEW.original_size,
        COALESCE((SELECT total_compressed_size FROM compression_stats WHERE table_name = 'memories'), 0) + NEW.compressed_size
    );
END;

-- Create views for monitoring
CREATE VIEW IF NOT EXISTS compression_summary AS
SELECT 
    table_name,
    total_original_size,
    total_compressed_size,
    ROUND(CAST(total_original_size AS REAL) / total_compressed_size, 2) as compression_ratio,
    ROUND((1.0 - CAST(total_compressed_size AS REAL) / total_original_size) * 100, 2) as space_saved_percent
FROM compression_stats
WHERE total_original_size > 0;

CREATE VIEW IF NOT EXISTS consciousness_timeline AS
SELECT 
    cs.timestamp,
    cs.self_awareness_level,
    cs.introspective_depth,
    COUNT(m.id) as memory_count
FROM consciousness_states cs
LEFT JOIN memories m ON cs.id = m.consciousness_state_id
GROUP BY cs.id
ORDER BY cs.timestamp DESC;