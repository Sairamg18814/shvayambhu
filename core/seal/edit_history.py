"""Edit history tracking system for SEAL architecture.

This module provides comprehensive tracking and analysis of all
self-editing operations for monitoring, debugging, and optimization.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Iterator
from dataclasses import dataclass, field, asdict
import time
import json
import sqlite3
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from .parameter_diff import ParameterDiff
from .edit_validation import ValidationResult


@dataclass
class EditEvent:
    """A comprehensive record of an edit event."""
    event_id: str
    timestamp: float
    event_type: str  # 'proposal', 'validation', 'application', 'rollback'
    
    # Edit details
    parameter_name: str
    diff_type: str
    magnitude: float
    
    # Context
    trigger_reason: str  # What triggered this edit
    performance_context: Dict[str, float] = field(default_factory=dict)
    model_state_hash: Optional[str] = None
    
    # Results
    validation_result: Optional[ValidationResult] = None
    application_success: bool = True
    rollback_reason: Optional[str] = None
    
    # Performance impact
    performance_before: Optional[Dict[str, float]] = None
    performance_after: Optional[Dict[str, float]] = None
    performance_delta: Optional[Dict[str, float]] = None
    
    # Metadata
    session_id: str = ""
    edit_sequence_number: int = 0
    related_edits: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class EditSession:
    """A session of related edits."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    description: str = ""
    
    # Session statistics
    total_edits: int = 0
    successful_edits: int = 0
    failed_edits: int = 0
    rolled_back_edits: int = 0
    
    # Performance tracking
    session_performance_start: Optional[Dict[str, float]] = None
    session_performance_end: Optional[Dict[str, float]] = None
    
    # Goals and outcomes
    target_metrics: Dict[str, float] = field(default_factory=dict)
    achieved_metrics: Dict[str, float] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        """Check if session is still active."""
        return self.end_time is None


class EditHistoryTracker:
    """Tracks and analyzes edit history."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("edit_history.db")
        self.current_session: Optional[EditSession] = None
        self.edit_sequence_number = 0
        
        # In-memory buffers for fast access
        self.recent_events: deque = deque(maxlen=1000)
        self.performance_timeline: List[Tuple[float, Dict[str, float]]] = []
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Edit events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS edit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    event_type TEXT,
                    parameter_name TEXT,
                    diff_type TEXT,
                    magnitude REAL,
                    trigger_reason TEXT,
                    validation_confidence REAL,
                    application_success INTEGER,
                    session_id TEXT,
                    edit_sequence_number INTEGER,
                    notes TEXT,
                    data_json TEXT
                )
            """)
            
            # Edit sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS edit_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time REAL,
                    end_time REAL,
                    description TEXT,
                    total_edits INTEGER,
                    successful_edits INTEGER,
                    failed_edits INTEGER,
                    rolled_back_edits INTEGER,
                    data_json TEXT
                )
            """)
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    timestamp REAL,
                    session_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    context TEXT
                )
            """)
            
            conn.commit()
    
    def start_session(
        self,
        description: str = "",
        target_metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """Start a new edit session."""
        if self.current_session and self.current_session.is_active():
            self.end_session()
        
        session_id = f"session_{int(time.time())}_{hash(description) % 10000:04d}"
        
        self.current_session = EditSession(
            session_id=session_id,
            start_time=time.time(),
            description=description,
            target_metrics=target_metrics or {}
        )
        
        self.edit_sequence_number = 0
        
        print(f"Started edit session: {session_id}")
        return session_id
    
    def end_session(self, achieved_metrics: Optional[Dict[str, float]] = None):
        """End the current edit session."""
        if not self.current_session:
            return
        
        self.current_session.end_time = time.time()
        if achieved_metrics:
            self.current_session.achieved_metrics = achieved_metrics
        
        # Save session to database
        self._save_session_to_db(self.current_session)
        
        print(f"Ended edit session: {self.current_session.session_id}")
        self.current_session = None
    
    def record_edit_proposal(
        self,
        parameter_name: str,
        diff: ParameterDiff,
        trigger_reason: str,
        performance_context: Optional[Dict[str, float]] = None
    ) -> str:
        """Record an edit proposal."""
        event_id = f"proposal_{int(time.time() * 1000)}_{hash(parameter_name) % 1000:03d}"
        
        event = EditEvent(
            event_id=event_id,
            timestamp=time.time(),
            event_type="proposal",
            parameter_name=parameter_name,
            diff_type=diff.diff_type,
            magnitude=diff.magnitude,
            trigger_reason=trigger_reason,
            performance_context=performance_context or {},
            session_id=self.current_session.session_id if self.current_session else "",
            edit_sequence_number=self.edit_sequence_number
        )
        
        self.recent_events.append(event)
        self._save_event_to_db(event)
        
        return event_id
    
    def record_edit_validation(
        self,
        event_id: str,
        validation_result: ValidationResult
    ):
        """Record edit validation results."""
        # Update the existing event or create validation event
        validation_event = EditEvent(
            event_id=f"validation_{event_id}",
            timestamp=time.time(),
            event_type="validation",
            parameter_name="",  # Will be filled from original event
            diff_type="",
            magnitude=0.0,
            trigger_reason="validation",
            validation_result=validation_result,
            session_id=self.current_session.session_id if self.current_session else ""
        )
        
        # Find and update the original proposal event
        for event in self.recent_events:
            if event.event_id == event_id:
                event.validation_result = validation_result
                break
        
        self.recent_events.append(validation_event)
        self._save_event_to_db(validation_event)
    
    def record_edit_application(
        self,
        event_id: str,
        success: bool,
        performance_before: Optional[Dict[str, float]] = None,
        performance_after: Optional[Dict[str, float]] = None
    ):
        """Record edit application results."""
        self.edit_sequence_number += 1
        
        # Calculate performance delta
        performance_delta = None
        if performance_before and performance_after:
            performance_delta = {
                key: performance_after.get(key, 0) - performance_before.get(key, 0)
                for key in set(performance_before.keys()) | set(performance_after.keys())
            }
        
        application_event = EditEvent(
            event_id=f"application_{event_id}",
            timestamp=time.time(),
            event_type="application",
            parameter_name="",  # Will be filled from original event
            diff_type="",
            magnitude=0.0,
            trigger_reason="application",
            application_success=success,
            performance_before=performance_before,
            performance_after=performance_after,
            performance_delta=performance_delta,
            session_id=self.current_session.session_id if self.current_session else "",
            edit_sequence_number=self.edit_sequence_number
        )
        
        # Update session statistics
        if self.current_session:
            self.current_session.total_edits += 1
            if success:
                self.current_session.successful_edits += 1
            else:
                self.current_session.failed_edits += 1
        
        # Update performance timeline
        if performance_after:
            self.performance_timeline.append((time.time(), performance_after))
        
        self.recent_events.append(application_event)
        self._save_event_to_db(application_event)
    
    def record_edit_rollback(
        self,
        original_event_id: str,
        rollback_reason: str
    ):
        """Record an edit rollback."""
        rollback_event = EditEvent(
            event_id=f"rollback_{original_event_id}",
            timestamp=time.time(),
            event_type="rollback",
            parameter_name="",
            diff_type="",
            magnitude=0.0,
            trigger_reason="rollback",
            rollback_reason=rollback_reason,
            session_id=self.current_session.session_id if self.current_session else ""
        )
        
        # Update session statistics
        if self.current_session:
            self.current_session.rolled_back_edits += 1
        
        self.recent_events.append(rollback_event)
        self._save_event_to_db(rollback_event)
    
    def _save_event_to_db(self, event: EditEvent):
        """Save event to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Prepare data
            validation_confidence = (event.validation_result.confidence 
                                   if event.validation_result else None)
            application_success = 1 if event.application_success else 0
            
            # Store complex data as JSON
            data_json = json.dumps({
                "performance_context": event.performance_context,
                "performance_before": event.performance_before,
                "performance_after": event.performance_after,
                "performance_delta": event.performance_delta,
                "validation_result": (asdict(event.validation_result) 
                                    if event.validation_result else None),
                "related_edits": event.related_edits
            })
            
            cursor.execute("""
                INSERT OR REPLACE INTO edit_events 
                (event_id, timestamp, event_type, parameter_name, diff_type, magnitude,
                 trigger_reason, validation_confidence, application_success, session_id,
                 edit_sequence_number, notes, data_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id, event.timestamp, event.event_type, event.parameter_name,
                event.diff_type, event.magnitude, event.trigger_reason,
                validation_confidence, application_success, event.session_id,
                event.edit_sequence_number, event.notes, data_json
            ))
            
            conn.commit()
    
    def _save_session_to_db(self, session: EditSession):
        """Save session to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            data_json = json.dumps({
                "target_metrics": session.target_metrics,
                "achieved_metrics": session.achieved_metrics,
                "session_performance_start": session.session_performance_start,
                "session_performance_end": session.session_performance_end
            })
            
            cursor.execute("""
                INSERT OR REPLACE INTO edit_sessions
                (session_id, start_time, end_time, description, total_edits,
                 successful_edits, failed_edits, rolled_back_edits, data_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id, session.start_time, session.end_time,
                session.description, session.total_edits, session.successful_edits,
                session.failed_edits, session.rolled_back_edits, data_json
            ))
            
            conn.commit()
    
    def get_edit_statistics(
        self,
        time_window_hours: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get comprehensive edit statistics."""
        cutoff_time = (time.time() - time_window_hours * 3600 
                      if time_window_hours else 0)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Basic statistics
            cursor.execute("""
                SELECT COUNT(*), AVG(magnitude), event_type, diff_type
                FROM edit_events 
                WHERE timestamp > ?
                GROUP BY event_type, diff_type
            """, (cutoff_time,))
            
            stats = {
                "total_events": 0,
                "events_by_type": defaultdict(int),
                "events_by_diff_type": defaultdict(int),
                "avg_magnitude_by_type": {},
                "success_rate": 0.0,
                "rollback_rate": 0.0
            }
            
            for count, avg_mag, event_type, diff_type in cursor.fetchall():
                stats["total_events"] += count
                stats["events_by_type"][event_type] += count
                stats["events_by_diff_type"][diff_type] += count
                stats["avg_magnitude_by_type"][f"{event_type}_{diff_type}"] = avg_mag
            
            # Success rate
            cursor.execute("""
                SELECT SUM(application_success), COUNT(*)
                FROM edit_events
                WHERE event_type = 'application' AND timestamp > ?
            """, (cutoff_time,))
            
            result = cursor.fetchone()
            if result and result[1] > 0:
                stats["success_rate"] = result[0] / result[1]
            
            # Rollback rate
            rollback_count = stats["events_by_type"]["rollback"]
            application_count = stats["events_by_type"]["application"]
            if application_count > 0:
                stats["rollback_rate"] = rollback_count / application_count
        
        return dict(stats)
    
    def get_performance_trends(
        self,
        metric_names: Optional[List[str]] = None,
        time_window_hours: float = 24.0
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Get performance trends over time."""
        cutoff_time = time.time() - time_window_hours * 3600
        
        trends = defaultdict(list)
        
        for timestamp, metrics in self.performance_timeline:
            if timestamp > cutoff_time:
                for metric_name, value in metrics.items():
                    if not metric_names or metric_name in metric_names:
                        trends[metric_name].append((timestamp, value))
        
        return dict(trends)
    
    def get_parameter_edit_frequency(self) -> Dict[str, int]:
        """Get frequency of edits per parameter."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT parameter_name, COUNT(*)
                FROM edit_events
                WHERE event_type IN ('proposal', 'application')
                GROUP BY parameter_name
                ORDER BY COUNT(*) DESC
            """)
            
            return dict(cursor.fetchall())
    
    def analyze_edit_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in edit behavior."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT timestamp, event_type, parameter_name, diff_type, magnitude,
                       trigger_reason, validation_confidence, application_success
                FROM edit_events
                ORDER BY timestamp
            """, conn)
        
        if df.empty:
            return {}
        
        patterns = {
            "most_edited_parameters": df[df['event_type'] == 'application']['parameter_name'].value_counts().head(10).to_dict(),
            "most_common_triggers": df['trigger_reason'].value_counts().head(10).to_dict(),
            "diff_type_distribution": df['diff_type'].value_counts().to_dict(),
            "hourly_activity": df.groupby(df['timestamp'].apply(
                lambda x: datetime.fromtimestamp(x).hour
            )).size().to_dict(),
            "validation_vs_success": df[df['validation_confidence'].notna()].groupby(
                pd.cut(df['validation_confidence'], bins=5)
            )['application_success'].mean().to_dict()
        }
        
        return patterns
    
    def export_edit_history(
        self,
        filepath: Path,
        format: str = "json",
        time_window_hours: Optional[float] = None
    ):
        """Export edit history to file."""
        cutoff_time = (time.time() - time_window_hours * 3600 
                      if time_window_hours else 0)
        
        with sqlite3.connect(self.db_path) as conn:
            if format == "json":
                # Export as JSON
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM edit_events WHERE timestamp > ?
                    ORDER BY timestamp
                """, (cutoff_time,))
                
                events = []
                for row in cursor.fetchall():
                    event_dict = dict(zip([col[0] for col in cursor.description], row))
                    # Parse JSON data
                    if event_dict['data_json']:
                        event_dict.update(json.loads(event_dict['data_json']))
                    del event_dict['data_json']
                    events.append(event_dict)
                
                with open(filepath, 'w') as f:
                    json.dump(events, f, indent=2)
            
            elif format == "csv":
                # Export as CSV
                df = pd.read_sql_query("""
                    SELECT timestamp, event_type, parameter_name, diff_type, magnitude,
                           trigger_reason, validation_confidence, application_success,
                           session_id, edit_sequence_number
                    FROM edit_events
                    WHERE timestamp > ?
                    ORDER BY timestamp
                """, conn, params=(cutoff_time,))
                
                df.to_csv(filepath, index=False)
    
    def create_edit_timeline_plot(
        self,
        save_path: Optional[Path] = None,
        time_window_hours: float = 24.0
    ):
        """Create a visual timeline of edit events."""
        cutoff_time = time.time() - time_window_hours * 3600
        
        # Get events from recent memory
        events = [e for e in self.recent_events if e.timestamp > cutoff_time]
        
        if not events:
            print("No events to plot")
            return
        
        # Prepare data
        timestamps = [datetime.fromtimestamp(e.timestamp) for e in events]
        event_types = [e.event_type for e in events]
        magnitudes = [e.magnitude for e in events]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Event timeline
        colors = {'proposal': 'blue', 'validation': 'orange', 
                 'application': 'green', 'rollback': 'red'}
        
        for event_type in colors:
            mask = [et == event_type for et in event_types]
            if any(mask):
                ax1.scatter([t for t, m in zip(timestamps, mask) if m],
                           [event_type] * sum(mask),
                           c=colors[event_type], alpha=0.7, s=50)
        
        ax1.set_ylabel('Event Type')
        ax1.set_title('Edit Event Timeline')
        ax1.grid(True, alpha=0.3)
        
        # Magnitude over time
        ax2.plot(timestamps, magnitudes, 'b-', alpha=0.7, linewidth=1)
        ax2.scatter(timestamps, magnitudes, c='blue', alpha=0.7, s=30)
        ax2.set_ylabel('Edit Magnitude')
        ax2.set_xlabel('Time')
        ax2.set_title('Edit Magnitude Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def get_recent_events(self, n: int = 10) -> List[EditEvent]:
        """Get the n most recent events."""
        return list(self.recent_events)[-n:]
    
    def search_events(
        self,
        parameter_name: Optional[str] = None,
        event_type: Optional[str] = None,
        trigger_reason: Optional[str] = None,
        min_timestamp: Optional[float] = None,
        max_timestamp: Optional[float] = None
    ) -> List[EditEvent]:
        """Search events by criteria."""
        filtered_events = []
        
        for event in self.recent_events:
            if parameter_name and event.parameter_name != parameter_name:
                continue
            if event_type and event.event_type != event_type:
                continue
            if trigger_reason and event.trigger_reason != trigger_reason:
                continue
            if min_timestamp and event.timestamp < min_timestamp:
                continue
            if max_timestamp and event.timestamp > max_timestamp:
                continue
            
            filtered_events.append(event)
        
        return filtered_events