"""Data versioning system for reproducible training.

This module implements version control for training data, enabling
reproducible experiments and tracking data lineage.
"""

import hashlib
import json
import time
import shutil
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
import sqlite3
import pickle
import gzip
from datetime import datetime
import numpy as np
from contextlib import contextmanager
import fcntl  # File locking for concurrent access


@dataclass
class DataVersion:
    """Represents a version of the dataset."""
    version_id: str
    parent_version: Optional[str]
    created_at: float
    created_by: str
    description: str
    
    # Data statistics
    num_documents: int
    total_bytes: int
    unique_tokens: int
    
    # Content hash
    content_hash: str
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Changes from parent
    additions: List[str] = field(default_factory=list)
    deletions: List[str] = field(default_factory=list)
    modifications: List[str] = field(default_factory=list)


@dataclass
class DataCommit:
    """Represents a commit in data version history."""
    commit_id: str
    version_id: str
    timestamp: float
    author: str
    message: str
    
    # Changes
    files_added: List[str] = field(default_factory=list)
    files_removed: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    
    # Statistics
    bytes_added: int = 0
    bytes_removed: int = 0
    
    # Parent commits (for merges)
    parents: List[str] = field(default_factory=list)


class DataVersionControl:
    """Version control system for training data."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.repo_path.mkdir(parents=True, exist_ok=True)
        
        # Repository structure
        self.versions_dir = self.repo_path / "versions"
        self.objects_dir = self.repo_path / "objects"
        self.refs_dir = self.repo_path / "refs"
        self.db_path = self.repo_path / "version_db.sqlite"
        
        # Create directories
        self.versions_dir.mkdir(exist_ok=True)
        self.objects_dir.mkdir(exist_ok=True)
        self.refs_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Current version
        self.current_version: Optional[str] = self._read_head()
    
    def _init_database(self):
        """Initialize SQLite database for metadata."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS versions (
                version_id TEXT PRIMARY KEY,
                parent_version TEXT,
                created_at REAL,
                created_by TEXT,
                description TEXT,
                num_documents INTEGER,
                total_bytes INTEGER,
                unique_tokens INTEGER,
                content_hash TEXT,
                metadata TEXT
            )
        """)
        
        # Commits table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS commits (
                commit_id TEXT PRIMARY KEY,
                version_id TEXT,
                timestamp REAL,
                author TEXT,
                message TEXT,
                parents TEXT,
                changes TEXT,
                FOREIGN KEY (version_id) REFERENCES versions(version_id)
            )
        """)
        
        # File tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT,
                file_size INTEGER,
                version_id TEXT,
                created_at REAL,
                FOREIGN KEY (version_id) REFERENCES versions(version_id)
            )
        """)
        
        # Indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_version_parent ON versions(parent_version)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_commit_version ON commits(version_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_version ON files(version_id)")
        
        conn.commit()
        conn.close()
    
    def _read_head(self) -> Optional[str]:
        """Read current HEAD version."""
        head_file = self.refs_dir / "HEAD"
        if head_file.exists():
            return head_file.read_text().strip()
        return None
    
    def _write_head(self, version_id: str):
        """Write HEAD version."""
        head_file = self.refs_dir / "HEAD"
        head_file.write_text(version_id)
        self.current_version = version_id
    
    @contextmanager
    def _lock_repo(self):
        """Lock repository for exclusive access."""
        lock_file = self.repo_path / ".lock"
        with open(lock_file, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    def init_version(
        self,
        description: str,
        author: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Initialize a new version."""
        version_id = self._generate_version_id()
        version_dir = self.versions_dir / version_id
        version_dir.mkdir(exist_ok=True)
        
        # Create data directories
        (version_dir / "raw").mkdir(exist_ok=True)
        (version_dir / "processed").mkdir(exist_ok=True)
        (version_dir / "metadata").mkdir(exist_ok=True)
        
        # Create version object
        version = DataVersion(
            version_id=version_id,
            parent_version=self.current_version,
            created_at=time.time(),
            created_by=author,
            description=description,
            num_documents=0,
            total_bytes=0,
            unique_tokens=0,
            content_hash="",
            metadata=metadata or {}
        )
        
        # Save version
        self._save_version(version)
        
        # Update HEAD
        self._write_head(version_id)
        
        return version_id
    
    def _generate_version_id(self) -> str:
        """Generate unique version ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_part = hashlib.sha256(
            f"{timestamp}{np.random.rand()}".encode()
        ).hexdigest()[:8]
        return f"v_{timestamp}_{random_part}"
    
    def _save_version(self, version: DataVersion):
        """Save version to database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO versions 
            (version_id, parent_version, created_at, created_by, description,
             num_documents, total_bytes, unique_tokens, content_hash, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            version.version_id,
            version.parent_version,
            version.created_at,
            version.created_by,
            version.description,
            version.num_documents,
            version.total_bytes,
            version.unique_tokens,
            version.content_hash,
            json.dumps(version.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def add_file(
        self,
        file_path: str,
        data: bytes,
        version_id: Optional[str] = None
    ) -> str:
        """Add file to version."""
        if version_id is None:
            version_id = self.current_version
        
        if version_id is None:
            raise ValueError("No version specified and no current version")
        
        # Compute file hash
        file_hash = hashlib.sha256(data).hexdigest()
        
        # Store in objects directory
        object_path = self.objects_dir / file_hash[:2] / file_hash[2:]
        object_path.parent.mkdir(exist_ok=True)
        
        if not object_path.exists():
            # Compress and store
            with gzip.open(object_path, 'wb') as f:
                f.write(data)
        
        # Link to version
        version_dir = self.versions_dir / version_id / "raw"
        link_path = version_dir / file_path
        link_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create symlink or copy
        if not link_path.exists():
            try:
                link_path.symlink_to(object_path)
            except OSError:
                # Fallback to copy if symlinks not supported
                shutil.copy2(object_path, link_path)
        
        # Update database
        self._track_file(file_hash, file_path, len(data), version_id)
        
        return file_hash
    
    def _track_file(
        self,
        file_hash: str,
        file_path: str,
        file_size: int,
        version_id: str
    ):
        """Track file in database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO files 
            (file_hash, file_path, file_size, version_id, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (file_hash, file_path, file_size, version_id, time.time()))
        
        conn.commit()
        conn.close()
    
    def commit(
        self,
        message: str,
        author: str = "system",
        version_id: Optional[str] = None
    ) -> str:
        """Commit changes to version."""
        if version_id is None:
            version_id = self.current_version
        
        if version_id is None:
            raise ValueError("No version to commit")
        
        with self._lock_repo():
            # Generate commit ID
            commit_id = hashlib.sha256(
                f"{version_id}{time.time()}{message}".encode()
            ).hexdigest()
            
            # Calculate changes
            changes = self._calculate_changes(version_id)
            
            # Create commit object
            commit = DataCommit(
                commit_id=commit_id,
                version_id=version_id,
                timestamp=time.time(),
                author=author,
                message=message,
                files_added=changes["added"],
                files_removed=changes["removed"],
                files_modified=changes["modified"],
                bytes_added=changes["bytes_added"],
                bytes_removed=changes["bytes_removed"]
            )
            
            # Save commit
            self._save_commit(commit)
            
            # Update version statistics
            self._update_version_stats(version_id)
            
            return commit_id
    
    def _calculate_changes(self, version_id: str) -> Dict[str, Any]:
        """Calculate changes in version."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Get current version files
        cursor.execute(
            "SELECT file_path, file_size FROM files WHERE version_id = ?",
            (version_id,)
        )
        current_files = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get parent version files if exists
        cursor.execute(
            "SELECT parent_version FROM versions WHERE version_id = ?",
            (version_id,)
        )
        parent_version = cursor.fetchone()
        
        if parent_version and parent_version[0]:
            cursor.execute(
                "SELECT file_path, file_size FROM files WHERE version_id = ?",
                (parent_version[0],)
            )
            parent_files = {row[0]: row[1] for row in cursor.fetchall()}
        else:
            parent_files = {}
        
        conn.close()
        
        # Calculate changes
        added = list(set(current_files) - set(parent_files))
        removed = list(set(parent_files) - set(current_files))
        modified = []
        
        bytes_added = sum(current_files[f] for f in added)
        bytes_removed = sum(parent_files[f] for f in removed)
        
        return {
            "added": added,
            "removed": removed,
            "modified": modified,
            "bytes_added": bytes_added,
            "bytes_removed": bytes_removed
        }
    
    def _save_commit(self, commit: DataCommit):
        """Save commit to database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        changes = {
            "files_added": commit.files_added,
            "files_removed": commit.files_removed,
            "files_modified": commit.files_modified,
            "bytes_added": commit.bytes_added,
            "bytes_removed": commit.bytes_removed
        }
        
        cursor.execute("""
            INSERT INTO commits 
            (commit_id, version_id, timestamp, author, message, parents, changes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            commit.commit_id,
            commit.version_id,
            commit.timestamp,
            commit.author,
            commit.message,
            json.dumps(commit.parents),
            json.dumps(changes)
        ))
        
        conn.commit()
        conn.close()
    
    def _update_version_stats(self, version_id: str):
        """Update version statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Calculate statistics
        cursor.execute(
            "SELECT COUNT(*), SUM(file_size) FROM files WHERE version_id = ?",
            (version_id,)
        )
        num_files, total_bytes = cursor.fetchone()
        
        # Update version
        cursor.execute("""
            UPDATE versions 
            SET num_documents = ?, total_bytes = ?
            WHERE version_id = ?
        """, (num_files or 0, total_bytes or 0, version_id))
        
        conn.commit()
        conn.close()
    
    def checkout(self, version_id: str):
        """Checkout a specific version."""
        # Verify version exists
        if not (self.versions_dir / version_id).exists():
            raise ValueError(f"Version {version_id} not found")
        
        # Update HEAD
        self._write_head(version_id)
    
    def get_version(self, version_id: Optional[str] = None) -> DataVersion:
        """Get version information."""
        if version_id is None:
            version_id = self.current_version
        
        if version_id is None:
            raise ValueError("No version specified")
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM versions WHERE version_id = ?",
            (version_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise ValueError(f"Version {version_id} not found")
        
        return DataVersion(
            version_id=row[0],
            parent_version=row[1],
            created_at=row[2],
            created_by=row[3],
            description=row[4],
            num_documents=row[5],
            total_bytes=row[6],
            unique_tokens=row[7],
            content_hash=row[8],
            metadata=json.loads(row[9]) if row[9] else {}
        )
    
    def list_versions(self) -> List[DataVersion]:
        """List all versions."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT version_id FROM versions ORDER BY created_at DESC")
        version_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return [self.get_version(vid) for vid in version_ids]
    
    def get_history(
        self,
        version_id: Optional[str] = None,
        limit: int = 10
    ) -> List[DataCommit]:
        """Get commit history for version."""
        if version_id is None:
            version_id = self.current_version
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT commit_id, version_id, timestamp, author, message, parents, changes
            FROM commits 
            WHERE version_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (version_id, limit))
        
        commits = []
        for row in cursor.fetchall():
            changes = json.loads(row[6])
            commit = DataCommit(
                commit_id=row[0],
                version_id=row[1],
                timestamp=row[2],
                author=row[3],
                message=row[4],
                parents=json.loads(row[5]) if row[5] else [],
                files_added=changes.get("files_added", []),
                files_removed=changes.get("files_removed", []),
                files_modified=changes.get("files_modified", []),
                bytes_added=changes.get("bytes_added", 0),
                bytes_removed=changes.get("bytes_removed", 0)
            )
            commits.append(commit)
        
        conn.close()
        return commits
    
    def diff_versions(
        self,
        version1: str,
        version2: str
    ) -> Dict[str, List[str]]:
        """Compare two versions."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Get files for both versions
        cursor.execute(
            "SELECT file_path FROM files WHERE version_id = ?",
            (version1,)
        )
        files1 = set(row[0] for row in cursor.fetchall())
        
        cursor.execute(
            "SELECT file_path FROM files WHERE version_id = ?",
            (version2,)
        )
        files2 = set(row[0] for row in cursor.fetchall())
        
        conn.close()
        
        return {
            "added": list(files2 - files1),
            "removed": list(files1 - files2),
            "common": list(files1 & files2)
        }
    
    def create_branch(
        self,
        branch_name: str,
        from_version: Optional[str] = None
    ):
        """Create a new branch."""
        if from_version is None:
            from_version = self.current_version
        
        branch_file = self.refs_dir / f"branch_{branch_name}"
        if branch_file.exists():
            raise ValueError(f"Branch {branch_name} already exists")
        
        branch_file.write_text(from_version)
    
    def merge_versions(
        self,
        version1: str,
        version2: str,
        description: str,
        author: str = "system",
        strategy: str = "union"
    ) -> str:
        """Merge two versions."""
        # Create new version
        merged_version = self.init_version(
            description=description,
            author=author,
            metadata={
                "merge_from": [version1, version2],
                "merge_strategy": strategy
            }
        )
        
        # Get files from both versions
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Files from version1
        cursor.execute(
            "SELECT file_path, file_hash FROM files WHERE version_id = ?",
            (version1,)
        )
        files1 = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Files from version2
        cursor.execute(
            "SELECT file_path, file_hash FROM files WHERE version_id = ?",
            (version2,)
        )
        files2 = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        # Merge based on strategy
        if strategy == "union":
            # Include all files from both versions
            merged_files = {**files1, **files2}
        elif strategy == "intersection":
            # Include only common files
            common_paths = set(files1.keys()) & set(files2.keys())
            merged_files = {
                path: files1[path] for path in common_paths
            }
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")
        
        # Add files to merged version
        for file_path, file_hash in merged_files.items():
            # Link existing object
            object_path = self.objects_dir / file_hash[:2] / file_hash[2:]
            version_path = self.versions_dir / merged_version / "raw" / file_path
            version_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                version_path.symlink_to(object_path)
            except OSError:
                shutil.copy2(object_path, version_path)
        
        # Commit merge
        self.commit(
            message=f"Merge {version1} and {version2}",
            author=author,
            version_id=merged_version
        )
        
        return merged_version
    
    def export_version(
        self,
        version_id: Optional[str] = None,
        output_path: str = None,
        format: str = "archive"
    ):
        """Export version for distribution."""
        if version_id is None:
            version_id = self.current_version
        
        if output_path is None:
            output_path = f"{version_id}.tar.gz"
        
        version_dir = self.versions_dir / version_id
        
        if format == "archive":
            # Create tar.gz archive
            import tarfile
            with tarfile.open(output_path, "w:gz") as tar:
                tar.add(version_dir, arcname=version_id)
        elif format == "directory":
            # Copy to directory
            shutil.copytree(version_dir, output_path)
        else:
            raise ValueError(f"Unknown export format: {format}")
    
    def garbage_collect(self):
        """Remove unreferenced objects."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Get all referenced file hashes
        cursor.execute("SELECT DISTINCT file_hash FROM files")
        referenced_hashes = set(row[0] for row in cursor.fetchall())
        conn.close()
        
        # Remove unreferenced objects
        removed_count = 0
        removed_bytes = 0
        
        for obj_dir in self.objects_dir.iterdir():
            if obj_dir.is_dir():
                for obj_file in obj_dir.iterdir():
                    file_hash = obj_dir.name + obj_file.name
                    if file_hash not in referenced_hashes:
                        removed_bytes += obj_file.stat().st_size
                        obj_file.unlink()
                        removed_count += 1
        
        # Remove empty directories
        for obj_dir in self.objects_dir.iterdir():
            if obj_dir.is_dir() and not any(obj_dir.iterdir()):
                obj_dir.rmdir()
        
        return removed_count, removed_bytes