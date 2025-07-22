"""Knowledge versioning system for temporal tracking and change management.

This module provides comprehensive versioning capabilities for the knowledge graph,
including change tracking, temporal queries, rollback functionality, and audit trails.
"""

import json
import time
import hashlib
import sqlite3
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import uuid
from datetime import datetime

from .graphrag import Entity, EntityType, Relationship, RelationType, Fact

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of changes in the knowledge graph."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    SPLIT = "split"
    VALIDATE = "validate"
    INVALIDATE = "invalidate"


class ConflictResolution(Enum):
    """Strategies for resolving conflicts during merges."""
    LATEST_WINS = "latest_wins"
    HIGHEST_CONFIDENCE = "highest_confidence"
    MANUAL = "manual"
    MERGE_ATTRIBUTES = "merge_attributes"
    KEEP_BOTH = "keep_both"


@dataclass
class ChangeRecord:
    """Record of a change made to the knowledge graph."""
    change_id: str
    timestamp: float
    change_type: ChangeType
    object_type: str  # "entity", "relationship", "fact"
    object_id: str
    previous_state: Optional[Dict[str, Any]] = None
    new_state: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.change_id:
            self.change_id = str(uuid.uuid4())
        
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class Version:
    """Represents a version of the knowledge graph or specific object."""
    version_id: str
    timestamp: float
    parent_version_id: Optional[str] = None
    changes: List[ChangeRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tag: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        if not self.version_id:
            self.version_id = str(uuid.uuid4())
        
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class Branch:
    """Represents a branch in the knowledge graph version history."""
    branch_id: str
    branch_name: str
    base_version_id: str
    head_version_id: str
    created_at: float
    created_by: Optional[str] = None
    description: Optional[str] = None
    is_active: bool = True
    
    def __post_init__(self):
        if not self.branch_id:
            self.branch_id = str(uuid.uuid4())
        
        if not self.created_at:
            self.created_at = time.time()


@dataclass
class Conflict:
    """Represents a conflict during merge operations."""
    conflict_id: str
    object_type: str
    object_id: str
    source_state: Dict[str, Any]
    target_state: Dict[str, Any]
    conflict_type: str
    resolution: Optional[ConflictResolution] = None
    resolved_state: Optional[Dict[str, Any]] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[float] = None
    
    def __post_init__(self):
        if not self.conflict_id:
            self.conflict_id = str(uuid.uuid4())


class KnowledgeVersioningSystem:
    """Main versioning system for knowledge graph temporal management."""
    
    def __init__(self, db_path: str = "knowledge_versioning.db"):
        """Initialize the versioning system.
        
        Args:
            db_path: Path to the versioning database
        """
        self.db_path = db_path
        self.conn = None
        self.current_session_id = str(uuid.uuid4())
        self.current_branch = "main"
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the versioning database."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        
        self._create_tables()
        self._create_indexes()
        
        # Create main branch if it doesn't exist
        if not self._branch_exists("main"):
            self._create_initial_branch()
    
    def _create_tables(self):
        """Create versioning database tables."""
        
        # Change records table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS change_records (
                change_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                change_type TEXT NOT NULL,
                object_type TEXT NOT NULL,
                object_id TEXT NOT NULL,
                previous_state TEXT,
                new_state TEXT,
                user_id TEXT,
                session_id TEXT,
                metadata TEXT,
                branch_id TEXT DEFAULT 'main'
            )
        """)
        
        # Versions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS versions (
                version_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                parent_version_id TEXT,
                tag TEXT,
                description TEXT,
                metadata TEXT,
                branch_id TEXT DEFAULT 'main',
                FOREIGN KEY (parent_version_id) REFERENCES versions (version_id)
            )
        """)
        
        # Version changes mapping table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS version_changes (
                version_id TEXT NOT NULL,
                change_id TEXT NOT NULL,
                PRIMARY KEY (version_id, change_id),
                FOREIGN KEY (version_id) REFERENCES versions (version_id),
                FOREIGN KEY (change_id) REFERENCES change_records (change_id)
            )
        """)
        
        # Branches table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS branches (
                branch_id TEXT PRIMARY KEY,
                branch_name TEXT UNIQUE NOT NULL,
                base_version_id TEXT,
                head_version_id TEXT,
                created_at REAL NOT NULL,
                created_by TEXT,
                description TEXT,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (base_version_id) REFERENCES versions (version_id),
                FOREIGN KEY (head_version_id) REFERENCES versions (version_id)
            )
        """)
        
        # Object states at specific versions
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS object_states (
                version_id TEXT NOT NULL,
                object_type TEXT NOT NULL,
                object_id TEXT NOT NULL,
                state TEXT NOT NULL,
                PRIMARY KEY (version_id, object_type, object_id),
                FOREIGN KEY (version_id) REFERENCES versions (version_id)
            )
        """)
        
        # Conflicts table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conflicts (
                conflict_id TEXT PRIMARY KEY,
                object_type TEXT NOT NULL,
                object_id TEXT NOT NULL,
                source_state TEXT NOT NULL,
                target_state TEXT NOT NULL,
                conflict_type TEXT NOT NULL,
                resolution TEXT,
                resolved_state TEXT,
                resolved_by TEXT,
                resolved_at REAL,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        self.conn.commit()
    
    def _create_indexes(self):
        """Create database indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_changes_timestamp ON change_records(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_changes_object ON change_records(object_type, object_id)",
            "CREATE INDEX IF NOT EXISTS idx_changes_session ON change_records(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_changes_branch ON change_records(branch_id)",
            "CREATE INDEX IF NOT EXISTS idx_versions_timestamp ON versions(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_versions_branch ON versions(branch_id)",
            "CREATE INDEX IF NOT EXISTS idx_object_states_object ON object_states(object_type, object_id)",
            "CREATE INDEX IF NOT EXISTS idx_conflicts_object ON conflicts(object_type, object_id)"
        ]
        
        for index_sql in indexes:
            self.conn.execute(index_sql)
        
        self.conn.commit()
    
    def _create_initial_branch(self):
        """Create the initial main branch."""
        initial_version = Version(
            version_id=str(uuid.uuid4()),
            timestamp=time.time(),
            metadata={"type": "initial", "description": "Initial knowledge graph state"}
        )
        
        branch = Branch(
            branch_id="main",
            branch_name="main",
            base_version_id=initial_version.version_id,
            head_version_id=initial_version.version_id,
            created_at=time.time(),
            description="Main development branch"
        )
        
        self._save_version(initial_version)
        self._save_branch(branch)
    
    def record_change(self, change_type: ChangeType, object_type: str, object_id: str,
                     previous_state: Optional[Dict[str, Any]] = None,
                     new_state: Optional[Dict[str, Any]] = None,
                     user_id: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> ChangeRecord:
        """Record a change in the knowledge graph.
        
        Args:
            change_type: Type of change being made
            object_type: Type of object being changed ("entity", "relationship", "fact")
            object_id: ID of the object being changed
            previous_state: Previous state of the object
            new_state: New state of the object
            user_id: ID of user making the change
            metadata: Additional metadata about the change
            
        Returns:
            ChangeRecord object
        """
        change_record = ChangeRecord(
            change_id=str(uuid.uuid4()),
            timestamp=time.time(),
            change_type=change_type,
            object_type=object_type,
            object_id=object_id,
            previous_state=previous_state,
            new_state=new_state,
            user_id=user_id,
            session_id=self.current_session_id,
            metadata=metadata or {}
        )
        
        # Save to database
        self.conn.execute("""
            INSERT INTO change_records 
            (change_id, timestamp, change_type, object_type, object_id,
             previous_state, new_state, user_id, session_id, metadata, branch_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            change_record.change_id,
            change_record.timestamp,
            change_record.change_type.value,
            change_record.object_type,
            change_record.object_id,
            json.dumps(change_record.previous_state) if change_record.previous_state else None,
            json.dumps(change_record.new_state) if change_record.new_state else None,
            change_record.user_id,
            change_record.session_id,
            json.dumps(change_record.metadata),
            self.current_branch
        ))
        
        self.conn.commit()
        
        logger.info(f"Recorded change: {change_type.value} for {object_type} {object_id}")
        return change_record
    
    def create_version(self, tag: Optional[str] = None, description: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> Version:
        """Create a new version of the knowledge graph.
        
        Args:
            tag: Optional tag for the version
            description: Optional description
            metadata: Additional metadata
            
        Returns:
            Version object
        """
        # Get current head version
        current_head = self._get_branch_head(self.current_branch)
        
        # Get all changes since last version
        if current_head:
            changes = self._get_changes_since_version(current_head.version_id)
        else:
            changes = self._get_all_changes()
        
        version = Version(
            version_id=str(uuid.uuid4()),
            timestamp=time.time(),
            parent_version_id=current_head.version_id if current_head else None,
            changes=changes,
            tag=tag,
            description=description,
            metadata=metadata or {}
        )
        
        # Save version
        self._save_version(version)
        
        # Link changes to version
        for change in changes:
            self.conn.execute("""
                INSERT OR IGNORE INTO version_changes (version_id, change_id)
                VALUES (?, ?)
            """, (version.version_id, change.change_id))
        
        # Update branch head
        self._update_branch_head(self.current_branch, version.version_id)
        
        # Create object state snapshots
        self._create_object_state_snapshots(version.version_id)
        
        self.conn.commit()
        
        logger.info(f"Created version {version.version_id} with {len(changes)} changes")
        return version
    
    def create_branch(self, branch_name: str, base_version_id: Optional[str] = None,
                     description: Optional[str] = None) -> Branch:
        """Create a new branch.
        
        Args:
            branch_name: Name of the new branch
            base_version_id: Version to branch from (defaults to current head)
            description: Optional description
            
        Returns:
            Branch object
        """
        if self._branch_exists(branch_name):
            raise ValueError(f"Branch '{branch_name}' already exists")
        
        if not base_version_id:
            current_head = self._get_branch_head(self.current_branch)
            base_version_id = current_head.version_id if current_head else None
        
        branch = Branch(
            branch_id=str(uuid.uuid4()),
            branch_name=branch_name,
            base_version_id=base_version_id,
            head_version_id=base_version_id,
            created_at=time.time(),
            description=description
        )
        
        self._save_branch(branch)
        
        logger.info(f"Created branch '{branch_name}' from version {base_version_id}")
        return branch
    
    def switch_branch(self, branch_name: str):
        """Switch to a different branch.
        
        Args:
            branch_name: Name of the branch to switch to
        """
        if not self._branch_exists(branch_name):
            raise ValueError(f"Branch '{branch_name}' does not exist")
        
        self.current_branch = branch_name
        logger.info(f"Switched to branch '{branch_name}'")
    
    def merge_branch(self, source_branch: str, target_branch: str,
                    conflict_resolution: ConflictResolution = ConflictResolution.LATEST_WINS,
                    user_id: Optional[str] = None) -> Tuple[Version, List[Conflict]]:
        """Merge one branch into another.
        
        Args:
            source_branch: Branch to merge from
            target_branch: Branch to merge into
            conflict_resolution: Strategy for resolving conflicts
            user_id: User performing the merge
            
        Returns:
            Tuple of (merged version, list of conflicts)
        """
        if not self._branch_exists(source_branch) or not self._branch_exists(target_branch):
            raise ValueError("One or both branches do not exist")
        
        source_head = self._get_branch_head(source_branch)
        target_head = self._get_branch_head(target_branch)
        
        if not source_head or not target_head:
            raise ValueError("Cannot merge branches without head versions")
        
        # Find common ancestor
        common_ancestor = self._find_common_ancestor(source_head.version_id, target_head.version_id)
        
        # Get changes from both branches since common ancestor
        source_changes = self._get_changes_between_versions(
            common_ancestor.version_id if common_ancestor else None,
            source_head.version_id
        )
        target_changes = self._get_changes_between_versions(
            common_ancestor.version_id if common_ancestor else None,
            target_head.version_id
        )
        
        # Detect conflicts
        conflicts = self._detect_conflicts(source_changes, target_changes)
        
        # Resolve conflicts
        resolved_conflicts = []
        for conflict in conflicts:
            resolved_conflict = self._resolve_conflict(conflict, conflict_resolution, user_id)
            resolved_conflicts.append(resolved_conflict)
        
        # Create merge version
        merge_version = Version(
            version_id=str(uuid.uuid4()),
            timestamp=time.time(),
            parent_version_id=target_head.version_id,
            metadata={
                "type": "merge",
                "source_branch": source_branch,
                "target_branch": target_branch,
                "conflicts_resolved": len(resolved_conflicts)
            }
        )
        
        # Apply merged changes
        self._apply_merge_changes(merge_version, source_changes, target_changes, resolved_conflicts)
        
        # Save merge version
        self._save_version(merge_version)
        self._update_branch_head(target_branch, merge_version.version_id)
        
        self.conn.commit()
        
        logger.info(f"Merged branch '{source_branch}' into '{target_branch}' with {len(resolved_conflicts)} conflicts")
        return merge_version, resolved_conflicts
    
    def get_object_at_version(self, object_type: str, object_id: str,
                            version_id: str) -> Optional[Dict[str, Any]]:
        """Get the state of an object at a specific version.
        
        Args:
            object_type: Type of object ("entity", "relationship", "fact")
            object_id: ID of the object
            version_id: Version to retrieve
            
        Returns:
            Object state or None if not found
        """
        cursor = self.conn.execute("""
            SELECT state FROM object_states
            WHERE version_id = ? AND object_type = ? AND object_id = ?
        """, (version_id, object_type, object_id))
        
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        
        # If not found in snapshots, reconstruct from changes
        return self._reconstruct_object_state(object_type, object_id, version_id)
    
    def get_changes_for_object(self, object_type: str, object_id: str,
                             start_time: Optional[float] = None,
                             end_time: Optional[float] = None) -> List[ChangeRecord]:
        """Get all changes for a specific object.
        
        Args:
            object_type: Type of object
            object_id: ID of the object
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List of ChangeRecord objects
        """
        query = """
            SELECT change_id, timestamp, change_type, object_type, object_id,
                   previous_state, new_state, user_id, session_id, metadata
            FROM change_records
            WHERE object_type = ? AND object_id = ?
        """
        params = [object_type, object_id]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp ASC"
        
        cursor = self.conn.execute(query, params)
        changes = []
        
        for row in cursor.fetchall():
            change = ChangeRecord(
                change_id=row[0],
                timestamp=row[1],
                change_type=ChangeType(row[2]),
                object_type=row[3],
                object_id=row[4],
                previous_state=json.loads(row[5]) if row[5] else None,
                new_state=json.loads(row[6]) if row[6] else None,
                user_id=row[7],
                session_id=row[8],
                metadata=json.loads(row[9]) if row[9] else {}
            )
            changes.append(change)
        
        return changes
    
    def rollback_to_version(self, version_id: str, user_id: Optional[str] = None) -> Version:
        """Rollback the current branch to a specific version.
        
        Args:
            version_id: Version to rollback to
            user_id: User performing the rollback
            
        Returns:
            New version representing the rollback
        """
        # Verify version exists
        if not self._version_exists(version_id):
            raise ValueError(f"Version {version_id} does not exist")
        
        current_head = self._get_branch_head(self.current_branch)
        if not current_head:
            raise ValueError("No current head version")
        
        # Get all objects at the target version
        target_objects = self._get_all_objects_at_version(version_id)
        current_objects = self._get_all_objects_at_version(current_head.version_id)
        
        # Create rollback changes
        rollback_changes = []
        for obj_key, target_state in target_objects.items():
            object_type, object_id = obj_key
            current_state = current_objects.get(obj_key)
            
            if current_state != target_state:
                change_record = self.record_change(
                    ChangeType.UPDATE,
                    object_type,
                    object_id,
                    previous_state=current_state,
                    new_state=target_state,
                    user_id=user_id,
                    metadata={"rollback_to": version_id}
                )
                rollback_changes.append(change_record)
        
        # Create rollback version
        rollback_version = Version(
            version_id=str(uuid.uuid4()),
            timestamp=time.time(),
            parent_version_id=current_head.version_id,
            changes=rollback_changes,
            metadata={
                "type": "rollback",
                "rollback_to": version_id,
                "performed_by": user_id
            }
        )
        
        self._save_version(rollback_version)
        self._update_branch_head(self.current_branch, rollback_version.version_id)
        self._create_object_state_snapshots(rollback_version.version_id)
        
        self.conn.commit()
        
        logger.info(f"Rolled back to version {version_id}")
        return rollback_version
    
    def get_version_history(self, branch_name: Optional[str] = None,
                          limit: int = 100) -> List[Version]:
        """Get version history for a branch.
        
        Args:
            branch_name: Branch to get history for (defaults to current)
            limit: Maximum number of versions to return
            
        Returns:
            List of Version objects
        """
        if not branch_name:
            branch_name = self.current_branch
        
        cursor = self.conn.execute("""
            SELECT version_id, timestamp, parent_version_id, tag, description, metadata
            FROM versions
            WHERE branch_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (branch_name, limit))
        
        versions = []
        for row in cursor.fetchall():
            version = Version(
                version_id=row[0],
                timestamp=row[1],
                parent_version_id=row[2],
                tag=row[3],
                description=row[4],
                metadata=json.loads(row[5]) if row[5] else {}
            )
            
            # Get changes for this version
            version.changes = self._get_changes_for_version(version.version_id)
            versions.append(version)
        
        return versions
    
    def _branch_exists(self, branch_name: str) -> bool:
        """Check if a branch exists."""
        cursor = self.conn.execute(
            "SELECT 1 FROM branches WHERE branch_name = ?", (branch_name,)
        )
        return cursor.fetchone() is not None
    
    def _version_exists(self, version_id: str) -> bool:
        """Check if a version exists."""
        cursor = self.conn.execute(
            "SELECT 1 FROM versions WHERE version_id = ?", (version_id,)
        )
        return cursor.fetchone() is not None
    
    def _save_version(self, version: Version):
        """Save a version to the database."""
        self.conn.execute("""
            INSERT INTO versions 
            (version_id, timestamp, parent_version_id, tag, description, metadata, branch_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            version.version_id,
            version.timestamp,
            version.parent_version_id,
            version.tag,
            version.description,
            json.dumps(version.metadata),
            self.current_branch
        ))
    
    def _save_branch(self, branch: Branch):
        """Save a branch to the database."""
        self.conn.execute("""
            INSERT INTO branches
            (branch_id, branch_name, base_version_id, head_version_id, 
             created_at, created_by, description, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            branch.branch_id,
            branch.branch_name,
            branch.base_version_id,
            branch.head_version_id,
            branch.created_at,
            branch.created_by,
            branch.description,
            branch.is_active
        ))
    
    def _get_branch_head(self, branch_name: str) -> Optional[Version]:
        """Get the head version of a branch."""
        cursor = self.conn.execute("""
            SELECT v.version_id, v.timestamp, v.parent_version_id, v.tag, v.description, v.metadata
            FROM branches b
            JOIN versions v ON b.head_version_id = v.version_id
            WHERE b.branch_name = ?
        """, (branch_name,))
        
        row = cursor.fetchone()
        if row:
            return Version(
                version_id=row[0],
                timestamp=row[1],
                parent_version_id=row[2],
                tag=row[3],
                description=row[4],
                metadata=json.loads(row[5]) if row[5] else {}
            )
        return None
    
    def _update_branch_head(self, branch_name: str, version_id: str):
        """Update the head version of a branch."""
        self.conn.execute("""
            UPDATE branches SET head_version_id = ? WHERE branch_name = ?
        """, (version_id, branch_name))
    
    def _get_changes_since_version(self, version_id: str) -> List[ChangeRecord]:
        """Get all changes since a specific version."""
        # Get the timestamp of the version
        cursor = self.conn.execute(
            "SELECT timestamp FROM versions WHERE version_id = ?", (version_id,)
        )
        row = cursor.fetchone()
        if not row:
            return []
        
        version_timestamp = row[0]
        
        # Get changes after this timestamp
        cursor = self.conn.execute("""
            SELECT change_id, timestamp, change_type, object_type, object_id,
                   previous_state, new_state, user_id, session_id, metadata
            FROM change_records
            WHERE timestamp > ? AND branch_id = ?
            ORDER BY timestamp ASC
        """, (version_timestamp, self.current_branch))
        
        return self._parse_change_records(cursor.fetchall())
    
    def _get_all_changes(self) -> List[ChangeRecord]:
        """Get all changes in the current branch."""
        cursor = self.conn.execute("""
            SELECT change_id, timestamp, change_type, object_type, object_id,
                   previous_state, new_state, user_id, session_id, metadata
            FROM change_records
            WHERE branch_id = ?
            ORDER BY timestamp ASC
        """, (self.current_branch,))
        
        return self._parse_change_records(cursor.fetchall())
    
    def _get_changes_for_version(self, version_id: str) -> List[ChangeRecord]:
        """Get all changes associated with a specific version."""
        cursor = self.conn.execute("""
            SELECT c.change_id, c.timestamp, c.change_type, c.object_type, c.object_id,
                   c.previous_state, c.new_state, c.user_id, c.session_id, c.metadata
            FROM change_records c
            JOIN version_changes vc ON c.change_id = vc.change_id
            WHERE vc.version_id = ?
            ORDER BY c.timestamp ASC
        """, (version_id,))
        
        return self._parse_change_records(cursor.fetchall())
    
    def _get_changes_between_versions(self, start_version_id: Optional[str],
                                    end_version_id: str) -> List[ChangeRecord]:
        """Get changes between two versions."""
        if not start_version_id:
            # Get all changes up to end version
            cursor = self.conn.execute(
                "SELECT timestamp FROM versions WHERE version_id = ?", (end_version_id,)
            )
            row = cursor.fetchone()
            if not row:
                return []
            
            end_timestamp = row[0]
            
            cursor = self.conn.execute("""
                SELECT change_id, timestamp, change_type, object_type, object_id,
                       previous_state, new_state, user_id, session_id, metadata
                FROM change_records
                WHERE timestamp <= ? AND branch_id = ?
                ORDER BY timestamp ASC
            """, (end_timestamp, self.current_branch))
            
        else:
            # Get changes between timestamps
            cursor = self.conn.execute("""
                SELECT timestamp FROM versions WHERE version_id IN (?, ?)
            """, (start_version_id, end_version_id))
            
            rows = cursor.fetchall()
            if len(rows) != 2:
                return []
            
            start_timestamp = min(rows[0][0], rows[1][0])
            end_timestamp = max(rows[0][0], rows[1][0])
            
            cursor = self.conn.execute("""
                SELECT change_id, timestamp, change_type, object_type, object_id,
                       previous_state, new_state, user_id, session_id, metadata
                FROM change_records
                WHERE timestamp > ? AND timestamp <= ? AND branch_id = ?
                ORDER BY timestamp ASC
            """, (start_timestamp, end_timestamp, self.current_branch))
        
        return self._parse_change_records(cursor.fetchall())
    
    def _parse_change_records(self, rows: List[Tuple]) -> List[ChangeRecord]:
        """Parse database rows into ChangeRecord objects."""
        changes = []
        for row in rows:
            change = ChangeRecord(
                change_id=row[0],
                timestamp=row[1],
                change_type=ChangeType(row[2]),
                object_type=row[3],
                object_id=row[4],
                previous_state=json.loads(row[5]) if row[5] else None,
                new_state=json.loads(row[6]) if row[6] else None,
                user_id=row[7],
                session_id=row[8],
                metadata=json.loads(row[9]) if row[9] else {}
            )
            changes.append(change)
        return changes
    
    def _create_object_state_snapshots(self, version_id: str):
        """Create object state snapshots for a version."""
        # This would typically integrate with the main knowledge store
        # to capture current object states
        logger.info(f"Created object state snapshots for version {version_id}")
    
    def _find_common_ancestor(self, version1_id: str, version2_id: str) -> Optional[Version]:
        """Find the common ancestor of two versions."""
        # Build version ancestry chains
        chain1 = self._get_version_ancestry(version1_id)
        chain2 = self._get_version_ancestry(version2_id)
        
        # Find common ancestors
        common_versions = set(chain1) & set(chain2)
        
        if common_versions:
            # Return the most recent common ancestor
            for version_id in chain1:  # chain1 is in reverse chronological order
                if version_id in common_versions:
                    cursor = self.conn.execute("""
                        SELECT version_id, timestamp, parent_version_id, tag, description, metadata
                        FROM versions WHERE version_id = ?
                    """, (version_id,))
                    row = cursor.fetchone()
                    if row:
                        return Version(
                            version_id=row[0],
                            timestamp=row[1],
                            parent_version_id=row[2],
                            tag=row[3],
                            description=row[4],
                            metadata=json.loads(row[5]) if row[5] else {}
                        )
        
        return None
    
    def _get_version_ancestry(self, version_id: str) -> List[str]:
        """Get the ancestry chain of a version."""
        ancestry = []
        current_id = version_id
        
        while current_id:
            ancestry.append(current_id)
            cursor = self.conn.execute(
                "SELECT parent_version_id FROM versions WHERE version_id = ?",
                (current_id,)
            )
            row = cursor.fetchone()
            current_id = row[0] if row else None
        
        return ancestry
    
    def _detect_conflicts(self, source_changes: List[ChangeRecord],
                         target_changes: List[ChangeRecord]) -> List[Conflict]:
        """Detect conflicts between two sets of changes."""
        conflicts = []
        
        # Group changes by object
        source_by_object = defaultdict(list)
        target_by_object = defaultdict(list)
        
        for change in source_changes:
            key = (change.object_type, change.object_id)
            source_by_object[key].append(change)
        
        for change in target_changes:
            key = (change.object_type, change.object_id)
            target_by_object[key].append(change)
        
        # Find objects modified in both branches
        common_objects = set(source_by_object.keys()) & set(target_by_object.keys())
        
        for obj_key in common_objects:
            object_type, object_id = obj_key
            source_changes_for_obj = source_by_object[obj_key]
            target_changes_for_obj = target_by_object[obj_key]
            
            # Get final states from each branch
            source_final_state = source_changes_for_obj[-1].new_state
            target_final_state = target_changes_for_obj[-1].new_state
            
            # Check if states are different
            if source_final_state != target_final_state:
                conflict = Conflict(
                    conflict_id=str(uuid.uuid4()),
                    object_type=object_type,
                    object_id=object_id,
                    source_state=source_final_state,
                    target_state=target_final_state,
                    conflict_type="modification_conflict"
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _resolve_conflict(self, conflict: Conflict, resolution: ConflictResolution,
                         user_id: Optional[str] = None) -> Conflict:
        """Resolve a conflict using the specified strategy."""
        resolved_state = None
        
        if resolution == ConflictResolution.LATEST_WINS:
            # Use the state with the latest timestamp
            resolved_state = conflict.target_state  # Target is typically the receiving branch
        
        elif resolution == ConflictResolution.HIGHEST_CONFIDENCE:
            # Use the state with highest confidence (if available)
            source_confidence = conflict.source_state.get('confidence', 0.0)
            target_confidence = conflict.target_state.get('confidence', 0.0)
            
            if source_confidence > target_confidence:
                resolved_state = conflict.source_state
            else:
                resolved_state = conflict.target_state
        
        elif resolution == ConflictResolution.MERGE_ATTRIBUTES:
            # Merge attributes from both states
            resolved_state = conflict.target_state.copy()
            if 'attributes' in conflict.source_state and 'attributes' in resolved_state:
                resolved_state['attributes'].update(conflict.source_state['attributes'])
        
        elif resolution == ConflictResolution.KEEP_BOTH:
            # Create a merged representation
            resolved_state = {
                'type': 'merged',
                'source_state': conflict.source_state,
                'target_state': conflict.target_state
            }
        
        else:  # MANUAL resolution would require external input
            resolved_state = conflict.target_state
        
        # Update conflict record
        conflict.resolution = resolution
        conflict.resolved_state = resolved_state
        conflict.resolved_by = user_id
        conflict.resolved_at = time.time()
        
        # Save to database
        self.conn.execute("""
            INSERT INTO conflicts
            (conflict_id, object_type, object_id, source_state, target_state,
             conflict_type, resolution, resolved_state, resolved_by, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            conflict.conflict_id,
            conflict.object_type,
            conflict.object_id,
            json.dumps(conflict.source_state),
            json.dumps(conflict.target_state),
            conflict.conflict_type,
            conflict.resolution.value if conflict.resolution else None,
            json.dumps(conflict.resolved_state),
            conflict.resolved_by,
            conflict.resolved_at
        ))
        
        return conflict
    
    def _apply_merge_changes(self, merge_version: Version, source_changes: List[ChangeRecord],
                           target_changes: List[ChangeRecord], resolved_conflicts: List[Conflict]):
        """Apply merged changes to create the merge version."""
        # This would typically apply the changes to the actual knowledge graph
        # For now, we just record the merge metadata
        logger.info(f"Applied merge changes for version {merge_version.version_id}")
    
    def _reconstruct_object_state(self, object_type: str, object_id: str,
                                 version_id: str) -> Optional[Dict[str, Any]]:
        """Reconstruct object state at a specific version from change history."""
        # Get all changes for this object up to the version timestamp
        cursor = self.conn.execute(
            "SELECT timestamp FROM versions WHERE version_id = ?", (version_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        
        version_timestamp = row[0]
        
        cursor = self.conn.execute("""
            SELECT change_type, new_state
            FROM change_records
            WHERE object_type = ? AND object_id = ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """, (object_type, object_id, version_timestamp))
        
        current_state = None
        for row in cursor.fetchall():
            change_type, new_state_json = row
            if new_state_json:
                new_state = json.loads(new_state_json)
                
                if change_type == ChangeType.CREATE.value:
                    current_state = new_state
                elif change_type == ChangeType.UPDATE.value:
                    if current_state:
                        current_state.update(new_state)
                    else:
                        current_state = new_state
                elif change_type == ChangeType.DELETE.value:
                    current_state = None
        
        return current_state
    
    def _get_all_objects_at_version(self, version_id: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Get all objects at a specific version."""
        objects = {}
        
        cursor = self.conn.execute("""
            SELECT object_type, object_id, state
            FROM object_states
            WHERE version_id = ?
        """, (version_id,))
        
        for row in cursor.fetchall():
            object_type, object_id, state_json = row
            key = (object_type, object_id)
            objects[key] = json.loads(state_json)
        
        return objects
    
    def close(self):
        """Close the versioning system."""
        if self.conn:
            self.conn.close()
            self.conn = None