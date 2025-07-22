"""Statistics tracking for training data and model performance.

This module provides comprehensive statistics tracking for monitoring
data quality, training progress, and model performance.
"""

import time
import json
import sqlite3
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import pandas as pd
from datetime import datetime
import threading
import queue

from ...core.blt.entropy import calculate_byte_entropy


@dataclass
class DataStatistics:
    """Statistics for a dataset."""
    # Basic counts
    num_samples: int = 0
    total_bytes: int = 0
    unique_bytes: int = 0
    
    # Length statistics
    min_length: int = float('inf')
    max_length: int = 0
    mean_length: float = 0.0
    std_length: float = 0.0
    
    # Content statistics
    mean_entropy: float = 0.0
    std_entropy: float = 0.0
    byte_distribution: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    # Language statistics
    language_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Quality metrics
    quality_scores: Dict[str, float] = field(default_factory=dict)
    
    # Time tracking
    processing_time: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrainingStatistics:
    """Statistics for training progress."""
    epoch: int = 0
    global_step: int = 0
    
    # Loss tracking
    loss_history: List[float] = field(default_factory=list)
    loss_components: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    # Performance metrics
    throughput_samples_per_sec: float = 0.0
    throughput_bytes_per_sec: float = 0.0
    
    # Learning rate
    learning_rate_history: List[float] = field(default_factory=list)
    
    # Gradient statistics
    gradient_norm_history: List[float] = field(default_factory=list)
    gradient_stats: Dict[str, float] = field(default_factory=dict)
    
    # Memory usage
    memory_usage_mb: List[float] = field(default_factory=list)
    
    # Validation metrics
    validation_metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    # Time tracking
    epoch_times: List[float] = field(default_factory=list)
    total_training_time: float = 0.0


@dataclass
class ModelStatistics:
    """Statistics for model performance."""
    # Model info
    model_name: str = ""
    num_parameters: int = 0
    model_size_mb: float = 0.0
    
    # Inference performance
    inference_latency_ms: List[float] = field(default_factory=list)
    first_token_latency_ms: List[float] = field(default_factory=list)
    tokens_per_second: List[float] = field(default_factory=list)
    
    # Quality metrics
    perplexity: List[float] = field(default_factory=list)
    accuracy: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    # Error analysis
    error_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    failure_cases: List[Dict[str, Any]] = field(default_factory=list)
    
    # Resource usage
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    
    # Optimization stats
    quantization_error: Optional[float] = None
    compression_ratio: Optional[float] = None


class StatisticsTracker:
    """Central statistics tracking system."""
    
    def __init__(self, output_dir: str, buffer_size: int = 1000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics storage
        self.data_stats = DataStatistics()
        self.training_stats = TrainingStatistics()
        self.model_stats = ModelStatistics()
        
        # Buffer for streaming updates
        self.buffer_size = buffer_size
        self.update_queue = queue.Queue()
        self.buffer_thread = threading.Thread(target=self._buffer_worker)
        self.buffer_thread.daemon = True
        self.buffer_thread.start()
        
        # Initialize database
        self.db_path = self.output_dir / "statistics.db"
        self._init_database()
        
        # Real-time tracking
        self.realtime_metrics = {
            "loss": deque(maxlen=100),
            "throughput": deque(maxlen=100),
            "memory": deque(maxlen=100)
        }
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Data statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                num_samples INTEGER,
                total_bytes INTEGER,
                mean_length REAL,
                mean_entropy REAL,
                processing_time REAL,
                metadata TEXT
            )
        """)
        
        # Training statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                epoch INTEGER,
                global_step INTEGER,
                loss REAL,
                learning_rate REAL,
                gradient_norm REAL,
                throughput REAL,
                memory_usage REAL,
                metadata TEXT
            )
        """)
        
        # Model statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                model_name TEXT,
                num_parameters INTEGER,
                inference_latency REAL,
                tokens_per_second REAL,
                perplexity REAL,
                memory_usage REAL,
                metadata TEXT
            )
        """)
        
        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_data_timestamp ON data_statistics(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_step ON training_statistics(global_step)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON model_statistics(model_name)")
        
        conn.commit()
        conn.close()
    
    def _buffer_worker(self):
        """Background worker for buffered updates."""
        buffer = []
        
        while True:
            try:
                # Get updates from queue
                update = self.update_queue.get(timeout=1.0)
                buffer.append(update)
                
                # Flush buffer if full
                if len(buffer) >= self.buffer_size:
                    self._flush_buffer(buffer)
                    buffer = []
                
            except queue.Empty:
                # Timeout - flush any pending updates
                if buffer:
                    self._flush_buffer(buffer)
                    buffer = []
    
    def _flush_buffer(self, buffer: List[Dict[str, Any]]):
        """Flush buffered updates to database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        for update in buffer:
            update_type = update.pop("type")
            
            if update_type == "data":
                cursor.execute("""
                    INSERT INTO data_statistics 
                    (timestamp, num_samples, total_bytes, mean_length, mean_entropy, processing_time, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    update["timestamp"],
                    update["num_samples"],
                    update["total_bytes"],
                    update["mean_length"],
                    update["mean_entropy"],
                    update["processing_time"],
                    json.dumps(update.get("metadata", {}))
                ))
            
            elif update_type == "training":
                cursor.execute("""
                    INSERT INTO training_statistics 
                    (timestamp, epoch, global_step, loss, learning_rate, gradient_norm, throughput, memory_usage, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    update["timestamp"],
                    update["epoch"],
                    update["global_step"],
                    update["loss"],
                    update["learning_rate"],
                    update["gradient_norm"],
                    update["throughput"],
                    update["memory_usage"],
                    json.dumps(update.get("metadata", {}))
                ))
            
            elif update_type == "model":
                cursor.execute("""
                    INSERT INTO model_statistics 
                    (timestamp, model_name, num_parameters, inference_latency, tokens_per_second, perplexity, memory_usage, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    update["timestamp"],
                    update["model_name"],
                    update["num_parameters"],
                    update["inference_latency"],
                    update["tokens_per_second"],
                    update["perplexity"],
                    update["memory_usage"],
                    json.dumps(update.get("metadata", {}))
                ))
        
        conn.commit()
        conn.close()
    
    def update_data_statistics(
        self,
        samples: List[bytes],
        processing_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update data statistics."""
        # Calculate statistics
        lengths = [len(s) for s in samples]
        entropies = [calculate_byte_entropy(s) for s in samples]
        
        # Update cumulative statistics
        self.data_stats.num_samples += len(samples)
        self.data_stats.total_bytes += sum(lengths)
        
        if lengths:
            self.data_stats.min_length = min(self.data_stats.min_length, min(lengths))
            self.data_stats.max_length = max(self.data_stats.max_length, max(lengths))
            self.data_stats.mean_length = np.mean(lengths)
            self.data_stats.std_length = np.std(lengths)
            self.data_stats.mean_entropy = np.mean(entropies)
            self.data_stats.std_entropy = np.std(entropies)
        
        # Update byte distribution
        for sample in samples:
            for byte in sample:
                self.data_stats.byte_distribution[byte] += 1
        
        self.data_stats.processing_time += processing_time
        
        # Queue update for database
        self.update_queue.put({
            "type": "data",
            "timestamp": time.time(),
            "num_samples": len(samples),
            "total_bytes": sum(lengths),
            "mean_length": self.data_stats.mean_length,
            "mean_entropy": self.data_stats.mean_entropy,
            "processing_time": processing_time,
            "metadata": metadata
        })
    
    def update_training_statistics(
        self,
        epoch: int,
        global_step: int,
        loss: float,
        learning_rate: float,
        gradient_norm: float,
        throughput: float,
        memory_usage: float,
        loss_components: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update training statistics."""
        # Update stats
        self.training_stats.epoch = epoch
        self.training_stats.global_step = global_step
        self.training_stats.loss_history.append(loss)
        self.training_stats.learning_rate_history.append(learning_rate)
        self.training_stats.gradient_norm_history.append(gradient_norm)
        self.training_stats.throughput_samples_per_sec = throughput
        self.training_stats.memory_usage_mb.append(memory_usage)
        
        # Update loss components
        if loss_components:
            for name, value in loss_components.items():
                self.training_stats.loss_components[name].append(value)
        
        # Update realtime metrics
        self.realtime_metrics["loss"].append((global_step, loss))
        self.realtime_metrics["throughput"].append((global_step, throughput))
        self.realtime_metrics["memory"].append((global_step, memory_usage))
        
        # Queue update for database
        self.update_queue.put({
            "type": "training",
            "timestamp": time.time(),
            "epoch": epoch,
            "global_step": global_step,
            "loss": loss,
            "learning_rate": learning_rate,
            "gradient_norm": gradient_norm,
            "throughput": throughput,
            "memory_usage": memory_usage,
            "metadata": metadata
        })
    
    def update_model_statistics(
        self,
        model_name: str,
        num_parameters: int,
        inference_latency: float,
        tokens_per_second: float,
        perplexity: float,
        memory_usage: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update model statistics."""
        # Update stats
        self.model_stats.model_name = model_name
        self.model_stats.num_parameters = num_parameters
        self.model_stats.inference_latency_ms.append(inference_latency)
        self.model_stats.tokens_per_second.append(tokens_per_second)
        self.model_stats.perplexity.append(perplexity)
        
        # Update memory tracking
        if memory_usage > self.model_stats.peak_memory_mb:
            self.model_stats.peak_memory_mb = memory_usage
        
        # Queue update for database
        self.update_queue.put({
            "type": "model",
            "timestamp": time.time(),
            "model_name": model_name,
            "num_parameters": num_parameters,
            "inference_latency": inference_latency,
            "tokens_per_second": tokens_per_second,
            "perplexity": perplexity,
            "memory_usage": memory_usage,
            "metadata": metadata
        })
    
    def add_validation_metric(
        self,
        metric_name: str,
        value: float,
        step: Optional[int] = None
    ):
        """Add validation metric."""
        self.training_stats.validation_metrics[metric_name].append(value)
    
    def add_error_case(
        self,
        error_type: str,
        details: Dict[str, Any]
    ):
        """Track error case."""
        self.model_stats.error_types[error_type] += 1
        self.model_stats.failure_cases.append({
            "type": error_type,
            "timestamp": time.time(),
            "details": details
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all statistics."""
        return {
            "data": {
                "num_samples": self.data_stats.num_samples,
                "total_bytes": self.data_stats.total_bytes,
                "mean_length": self.data_stats.mean_length,
                "mean_entropy": self.data_stats.mean_entropy,
                "processing_time": self.data_stats.processing_time
            },
            "training": {
                "epoch": self.training_stats.epoch,
                "global_step": self.training_stats.global_step,
                "current_loss": self.training_stats.loss_history[-1] if self.training_stats.loss_history else None,
                "throughput": self.training_stats.throughput_samples_per_sec,
                "total_time": self.training_stats.total_training_time
            },
            "model": {
                "name": self.model_stats.model_name,
                "parameters": self.model_stats.num_parameters,
                "avg_latency": np.mean(self.model_stats.inference_latency_ms) if self.model_stats.inference_latency_ms else None,
                "avg_tokens_per_sec": np.mean(self.model_stats.tokens_per_second) if self.model_stats.tokens_per_second else None,
                "current_perplexity": self.model_stats.perplexity[-1] if self.model_stats.perplexity else None
            }
        }
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curve
        if self.training_stats.loss_history:
            axes[0, 0].plot(self.training_stats.loss_history)
            axes[0, 0].set_title("Training Loss")
            axes[0, 0].set_xlabel("Step")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].set_yscale('log')
        
        # Learning rate
        if self.training_stats.learning_rate_history:
            axes[0, 1].plot(self.training_stats.learning_rate_history)
            axes[0, 1].set_title("Learning Rate")
            axes[0, 1].set_xlabel("Step")
            axes[0, 1].set_ylabel("LR")
        
        # Gradient norm
        if self.training_stats.gradient_norm_history:
            axes[1, 0].plot(self.training_stats.gradient_norm_history)
            axes[1, 0].set_title("Gradient Norm")
            axes[1, 0].set_xlabel("Step")
            axes[1, 0].set_ylabel("Norm")
        
        # Memory usage
        if self.training_stats.memory_usage_mb:
            axes[1, 1].plot(self.training_stats.memory_usage_mb)
            axes[1, 1].set_title("Memory Usage")
            axes[1, 1].set_xlabel("Step")
            axes[1, 1].set_ylabel("MB")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_data_distribution(self, save_path: Optional[str] = None):
        """Plot data distribution statistics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Byte distribution
        if self.data_stats.byte_distribution:
            bytes_list = list(range(256))
            counts = [self.data_stats.byte_distribution.get(b, 0) for b in bytes_list]
            axes[0, 0].bar(bytes_list, counts, width=1)
            axes[0, 0].set_title("Byte Distribution")
            axes[0, 0].set_xlabel("Byte Value")
            axes[0, 0].set_ylabel("Count")
            axes[0, 0].set_yscale('log')
        
        # Language distribution
        if self.data_stats.language_distribution:
            languages = list(self.data_stats.language_distribution.keys())
            counts = list(self.data_stats.language_distribution.values())
            axes[0, 1].pie(counts, labels=languages, autopct='%1.1f%%')
            axes[0, 1].set_title("Language Distribution")
        
        # Length distribution (simulated)
        axes[1, 0].hist(np.random.normal(self.data_stats.mean_length, self.data_stats.std_length, 1000), bins=50)
        axes[1, 0].set_title("Sample Length Distribution")
        axes[1, 0].set_xlabel("Length (bytes)")
        axes[1, 0].set_ylabel("Count")
        
        # Entropy distribution (simulated)
        axes[1, 1].hist(np.random.normal(self.data_stats.mean_entropy, self.data_stats.std_entropy, 1000), bins=50)
        axes[1, 1].set_title("Entropy Distribution")
        axes[1, 1].set_xlabel("Entropy")
        axes[1, 1].set_ylabel("Count")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "data_distribution.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def generate_report(self, report_path: Optional[str] = None):
        """Generate comprehensive statistics report."""
        if report_path is None:
            report_path = self.output_dir / "statistics_report.html"
        
        # Create HTML report
        html_content = f"""
        <html>
        <head>
            <title>Shvayambhu Training Statistics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
            </style>
        </head>
        <body>
            <h1>Shvayambhu Training Statistics Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Data Statistics</h2>
            <table>
                <tr><td>Total Samples</td><td class="metric">{self.data_stats.num_samples:,}</td></tr>
                <tr><td>Total Bytes</td><td class="metric">{self.data_stats.total_bytes:,}</td></tr>
                <tr><td>Mean Sample Length</td><td>{self.data_stats.mean_length:.1f} bytes</td></tr>
                <tr><td>Mean Entropy</td><td>{self.data_stats.mean_entropy:.3f}</td></tr>
                <tr><td>Processing Time</td><td>{self.data_stats.processing_time:.2f} seconds</td></tr>
            </table>
            
            <h2>Training Progress</h2>
            <table>
                <tr><td>Current Epoch</td><td class="metric">{self.training_stats.epoch}</td></tr>
                <tr><td>Global Step</td><td class="metric">{self.training_stats.global_step:,}</td></tr>
                <tr><td>Current Loss</td><td>{self.training_stats.loss_history[-1] if self.training_stats.loss_history else 'N/A':.4f}</td></tr>
                <tr><td>Throughput</td><td>{self.training_stats.throughput_samples_per_sec:.1f} samples/sec</td></tr>
                <tr><td>Total Training Time</td><td>{self.training_stats.total_training_time:.2f} seconds</td></tr>
            </table>
            
            <h2>Model Performance</h2>
            <table>
                <tr><td>Model Name</td><td>{self.model_stats.model_name}</td></tr>
                <tr><td>Parameters</td><td class="metric">{self.model_stats.num_parameters:,}</td></tr>
                <tr><td>Avg Inference Latency</td><td>{np.mean(self.model_stats.inference_latency_ms) if self.model_stats.inference_latency_ms else 0:.2f} ms</td></tr>
                <tr><td>Avg Tokens/Second</td><td>{np.mean(self.model_stats.tokens_per_second) if self.model_stats.tokens_per_second else 0:.1f}</td></tr>
                <tr><td>Peak Memory Usage</td><td>{self.model_stats.peak_memory_mb:.1f} MB</td></tr>
            </table>
            
            <h2>Visualizations</h2>
            <img src="training_curves.png" width="800">
            <img src="data_distribution.png" width="800">
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        # Generate plots
        self.plot_training_curves()
        self.plot_data_distribution()
    
    def export_to_csv(self, csv_dir: Optional[str] = None):
        """Export statistics to CSV files."""
        if csv_dir is None:
            csv_dir = self.output_dir / "csv_exports"
        
        csv_dir = Path(csv_dir)
        csv_dir.mkdir(exist_ok=True)
        
        # Export training statistics
        if self.training_stats.loss_history:
            training_df = pd.DataFrame({
                "step": range(len(self.training_stats.loss_history)),
                "loss": self.training_stats.loss_history,
                "learning_rate": self.training_stats.learning_rate_history[:len(self.training_stats.loss_history)],
                "gradient_norm": self.training_stats.gradient_norm_history[:len(self.training_stats.loss_history)]
            })
            training_df.to_csv(csv_dir / "training_stats.csv", index=False)
        
        # Export model statistics
        if self.model_stats.inference_latency_ms:
            model_df = pd.DataFrame({
                "inference_latency_ms": self.model_stats.inference_latency_ms,
                "tokens_per_second": self.model_stats.tokens_per_second[:len(self.model_stats.inference_latency_ms)],
                "perplexity": self.model_stats.perplexity[:len(self.model_stats.inference_latency_ms)]
            })
            model_df.to_csv(csv_dir / "model_stats.csv", index=False)
    
    def query_database(
        self,
        query: str,
        params: Optional[Tuple] = None
    ) -> List[Tuple]:
        """Query the statistics database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def close(self):
        """Close statistics tracker."""
        # Flush any pending updates
        self.update_queue.put(None)  # Sentinel to stop worker
        self.buffer_thread.join()
        
        # Generate final report
        self.generate_report()