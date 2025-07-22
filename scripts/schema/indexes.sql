-- Additional performance indexes for Shvayambhu

-- Full-text search indexes for better retrieval
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    content=memories,
    content_rowid=id
);

-- Trigger to keep FTS index updated
CREATE TRIGGER IF NOT EXISTS memories_fts_insert
AFTER INSERT ON memories
BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (NEW.id, '');
END;

-- Performance indexes for consciousness tracking
CREATE INDEX IF NOT EXISTS idx_consciousness_awareness_level 
ON consciousness_states(self_awareness_level);

CREATE INDEX IF NOT EXISTS idx_consciousness_introspection 
ON consciousness_states(introspective_depth);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_memories_type_timestamp 
ON memories(type, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_memories_consciousness 
ON memories(consciousness_state_id, timestamp DESC);

-- Knowledge graph performance indexes
CREATE INDEX IF NOT EXISTS idx_knowledge_confidence 
ON knowledge_nodes(confidence_score DESC);

CREATE INDEX IF NOT EXISTS idx_knowledge_updated 
ON knowledge_nodes(updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_edges_strength 
ON knowledge_edges(strength DESC);

-- Web cache optimization
CREATE INDEX IF NOT EXISTS idx_web_cache_expiry 
ON web_cache(expiry_time);

CREATE INDEX IF NOT EXISTS idx_web_cache_hash 
ON web_cache(content_hash);

-- Training experience indexes
CREATE INDEX IF NOT EXISTS idx_training_timestamp 
ON training_experiences(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_training_loss 
ON training_experiences(loss_value);

-- Model checkpoint indexes
CREATE INDEX IF NOT EXISTS idx_checkpoints_created 
ON model_checkpoints(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_checkpoints_consciousness 
ON model_checkpoints(consciousness_level DESC);

-- Conversation performance
CREATE INDEX IF NOT EXISTS idx_conversations_timestamp 
ON conversations(timestamp DESC);

-- Partial indexes for specific queries
CREATE INDEX IF NOT EXISTS idx_recent_memories 
ON memories(timestamp DESC) 
WHERE timestamp > datetime('now', '-7 days');

CREATE INDEX IF NOT EXISTS idx_high_confidence_knowledge 
ON knowledge_nodes(entity_name) 
WHERE confidence_score > 0.8;

-- Expression indexes for JSON fields
CREATE INDEX IF NOT EXISTS idx_consciousness_emotion_type
ON consciousness_states(json_extract(emotional_state, '$.primary_emotion'));

CREATE INDEX IF NOT EXISTS idx_memories_metadata_source
ON memories(json_extract(metadata, '$.source'));

-- Analyze tables for query optimizer
ANALYZE;

-- Update SQLite statistics
PRAGMA optimize;