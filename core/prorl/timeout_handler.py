"""Timeout handling system for ProRL reasoning operations.

This module provides comprehensive timeout management for reasoning sessions,
including graceful degradation, partial result preservation, and adaptive timing.
"""

import time
import threading
import signal
import functools
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import queue
from contextlib import contextmanager

from .reasoning_engine import ReasoningTrace, ReasoningStep, ReasoningStrategy


class TimeoutReason(Enum):
    """Reasons for timeout occurrence."""
    HARD_TIMEOUT = "hard_timeout"
    SOFT_TIMEOUT = "soft_timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PROGRESS_STAGNATION = "progress_stagnation"
    USER_INTERRUPTION = "user_interruption"
    SYSTEM_OVERLOAD = "system_overload"


class TimeoutStrategy(Enum):
    """Strategies for handling timeouts."""
    IMMEDIATE_STOP = "immediate_stop"
    GRACEFUL_FINISH = "graceful_finish"
    BEST_EFFORT = "best_effort"
    INCREMENTAL_EXTEND = "incremental_extend"
    ADAPTIVE_SWITCH = "adaptive_switch"


@dataclass
class TimeoutEvent:
    """Information about a timeout event."""
    timestamp: float
    reason: TimeoutReason
    strategy_used: TimeoutStrategy
    partial_result: Optional[Any]
    steps_completed: int
    time_elapsed: float
    recovery_attempted: bool
    recovery_successful: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeoutConfig:
    """Configuration for timeout handling."""
    # Basic timeouts
    hard_timeout_seconds: float = 300.0  # 5 minutes absolute max
    soft_timeout_seconds: float = 180.0  # 3 minutes soft limit
    step_timeout_seconds: float = 30.0   # Max time per step
    
    # Progressive timeouts
    enable_progressive_timeout: bool = True
    progress_check_interval: float = 10.0  # Check progress every 10 seconds
    min_progress_rate: float = 0.1  # Minimum progress per minute
    stagnation_timeout: float = 60.0  # Timeout if no progress for 1 minute
    
    # Timeout strategies
    default_strategy: TimeoutStrategy = TimeoutStrategy.GRACEFUL_FINISH
    enable_adaptive_timeout: bool = True
    timeout_extensions: int = 2  # Number of extensions allowed
    extension_factor: float = 0.5  # Multiply by this factor for extensions
    
    # Resource monitoring
    enable_resource_monitoring: bool = True
    max_memory_mb: float = 8192.0
    max_cpu_percent: float = 90.0
    
    # Recovery options
    enable_timeout_recovery: bool = True
    quick_recovery_timeout: float = 30.0
    enable_partial_results: bool = True


class TimeoutHandler:
    """Handles timeouts for reasoning operations with graceful degradation."""
    
    def __init__(self, config: TimeoutConfig):
        self.config = config
        self._active_operations: Dict[str, Dict[str, Any]] = {}
        self._timeout_history: List[TimeoutEvent] = []
        self._lock = threading.Lock()
        
        # Monitoring thread
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Resource monitoring
        self._resource_monitor_enabled = config.enable_resource_monitoring
        
        # Start monitoring
        if config.enable_resource_monitoring or config.enable_progressive_timeout:
            self._start_monitoring()
    
    def with_timeout(
        self,
        timeout_seconds: Optional[float] = None,
        strategy: Optional[TimeoutStrategy] = None,
        operation_id: Optional[str] = None
    ):
        """Decorator for timeout-protected operations."""
        
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute_with_timeout(
                    func, args, kwargs,
                    timeout_seconds=timeout_seconds,
                    strategy=strategy,
                    operation_id=operation_id
                )
            return wrapper
        return decorator
    
    def execute_with_timeout(
        self,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict[str, Any] = None,
        timeout_seconds: Optional[float] = None,
        strategy: Optional[TimeoutStrategy] = None,
        operation_id: Optional[str] = None
    ) -> Tuple[Any, Optional[TimeoutEvent]]:
        """Execute function with timeout protection."""
        
        kwargs = kwargs or {}
        timeout_seconds = timeout_seconds or self.config.soft_timeout_seconds
        strategy = strategy or self.config.default_strategy
        operation_id = operation_id or f"op_{int(time.time() * 1000)}"
        
        start_time = time.time()
        timeout_event = None
        
        # Register operation
        self._register_operation(operation_id, func.__name__, start_time, timeout_seconds)
        
        try:
            # Use ThreadPoolExecutor for timeout capability
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                
                try:
                    # Wait for result with timeout
                    result = future.result(timeout=timeout_seconds)
                    
                    # Check if we exceeded soft timeout but not hard timeout
                    elapsed = time.time() - start_time
                    if elapsed > self.config.soft_timeout_seconds:
                        timeout_event = TimeoutEvent(
                            timestamp=time.time(),
                            reason=TimeoutReason.SOFT_TIMEOUT,
                            strategy_used=strategy,
                            partial_result=result,
                            steps_completed=self._get_steps_completed(operation_id),
                            time_elapsed=elapsed,
                            recovery_attempted=False,
                            recovery_successful=True
                        )
                    
                    return result, timeout_event
                    
                except FutureTimeoutError:
                    # Handle timeout based on strategy
                    timeout_event = self._handle_timeout(
                        operation_id, TimeoutReason.HARD_TIMEOUT, strategy, future
                    )
                    
                    if timeout_event.partial_result is not None:
                        return timeout_event.partial_result, timeout_event
                    else:
                        raise TimeoutError(f"Operation {operation_id} timed out after {timeout_seconds}s")
                
        finally:
            # Unregister operation
            self._unregister_operation(operation_id)
            
            # Record timeout event if occurred
            if timeout_event:
                self._timeout_history.append(timeout_event)
    
    def monitor_reasoning_session(
        self,
        trace: ReasoningTrace,
        operation_id: str,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Optional[TimeoutEvent]:
        """Monitor a reasoning session for timeout conditions."""
        
        start_time = time.time()
        last_progress_time = start_time
        last_step_count = 0
        
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check hard timeout
            if elapsed > self.config.hard_timeout_seconds:
                return self._create_timeout_event(
                    TimeoutReason.HARD_TIMEOUT,
                    self.config.default_strategy,
                    operation_id,
                    trace
                )
            
            # Check soft timeout
            if elapsed > self.config.soft_timeout_seconds:
                # Try graceful finish
                if self.config.default_strategy == TimeoutStrategy.GRACEFUL_FINISH:
                    return self._attempt_graceful_finish(operation_id, trace)
            
            # Check progress stagnation
            if self.config.enable_progressive_timeout:
                current_steps = len(trace.steps)
                if current_steps > last_step_count:
                    last_progress_time = current_time
                    last_step_count = current_steps
                    
                    # Report progress
                    if progress_callback:
                        progress = min(current_steps / max(trace.total_steps, 1), 1.0)
                        progress_callback(progress)
                
                # Check stagnation
                time_since_progress = current_time - last_progress_time
                if time_since_progress > self.config.stagnation_timeout:
                    return self._create_timeout_event(
                        TimeoutReason.PROGRESS_STAGNATION,
                        TimeoutStrategy.BEST_EFFORT,
                        operation_id,
                        trace
                    )
            
            # Check resource constraints
            if self._resource_monitor_enabled:
                if self._check_resource_exhaustion():
                    return self._create_timeout_event(
                        TimeoutReason.RESOURCE_EXHAUSTION,
                        TimeoutStrategy.IMMEDIATE_STOP,
                        operation_id,
                        trace
                    )
            
            # Sleep before next check
            time.sleep(self.config.progress_check_interval)
            
            # Check if operation is still active
            if operation_id not in self._active_operations:
                break
        
        return None
    
    def request_timeout_extension(
        self,
        operation_id: str,
        additional_seconds: float,
        justification: str = ""
    ) -> bool:
        """Request additional time for an operation."""
        
        with self._lock:
            if operation_id not in self._active_operations:
                return False
            
            operation = self._active_operations[operation_id]
            extensions_used = operation.get("extensions_used", 0)
            
            if extensions_used >= self.config.timeout_extensions:
                return False
            
            # Grant extension
            operation["timeout_seconds"] += additional_seconds
            operation["extensions_used"] = extensions_used + 1
            operation["extension_justifications"] = operation.get("extension_justifications", [])
            operation["extension_justifications"].append(justification)
            
            return True
    
    def cancel_operation(self, operation_id: str, reason: str = "User cancellation") -> bool:
        """Cancel an active operation."""
        
        with self._lock:
            if operation_id not in self._active_operations:
                return False
            
            operation = self._active_operations[operation_id]
            operation["cancelled"] = True
            operation["cancellation_reason"] = reason
            
            # Create timeout event
            timeout_event = TimeoutEvent(
                timestamp=time.time(),
                reason=TimeoutReason.USER_INTERRUPTION,
                strategy_used=TimeoutStrategy.IMMEDIATE_STOP,
                partial_result=operation.get("partial_result"),
                steps_completed=operation.get("steps_completed", 0),
                time_elapsed=time.time() - operation["start_time"],
                recovery_attempted=False,
                recovery_successful=False,
                metadata={"cancellation_reason": reason}
            )
            
            self._timeout_history.append(timeout_event)
            return True
    
    def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active operation."""
        
        with self._lock:
            if operation_id not in self._active_operations:
                return None
            
            operation = self._active_operations[operation_id].copy()
            operation["elapsed_time"] = time.time() - operation["start_time"]
            operation["remaining_time"] = max(0, operation["timeout_seconds"] - operation["elapsed_time"])
            
            return operation
    
    def get_timeout_statistics(self) -> Dict[str, Any]:
        """Get comprehensive timeout statistics."""
        
        if not self._timeout_history:
            return {"message": "No timeout events recorded"}
        
        total_timeouts = len(self._timeout_history)
        reason_counts = {}
        strategy_counts = {}
        recovery_success_count = 0
        
        for event in self._timeout_history:
            reason_counts[event.reason.value] = reason_counts.get(event.reason.value, 0) + 1
            strategy_counts[event.strategy_used.value] = strategy_counts.get(event.strategy_used.value, 0) + 1
            
            if event.recovery_successful:
                recovery_success_count += 1
        
        avg_time_to_timeout = sum(e.time_elapsed for e in self._timeout_history) / total_timeouts
        partial_result_rate = sum(1 for e in self._timeout_history if e.partial_result is not None) / total_timeouts
        
        return {
            "total_timeouts": total_timeouts,
            "reason_distribution": reason_counts,
            "strategy_distribution": strategy_counts,
            "recovery_success_rate": recovery_success_count / total_timeouts,
            "partial_result_rate": partial_result_rate,
            "avg_time_to_timeout": avg_time_to_timeout,
            "active_operations": len(self._active_operations),
            "config": {
                "hard_timeout": self.config.hard_timeout_seconds,
                "soft_timeout": self.config.soft_timeout_seconds,
                "step_timeout": self.config.step_timeout_seconds
            }
        }
    
    def _register_operation(self, operation_id: str, function_name: str, start_time: float, timeout_seconds: float):
        """Register a new operation for monitoring."""
        
        with self._lock:
            self._active_operations[operation_id] = {
                "function_name": function_name,
                "start_time": start_time,
                "timeout_seconds": timeout_seconds,
                "steps_completed": 0,
                "extensions_used": 0,
                "cancelled": False,
                "partial_result": None
            }
    
    def _unregister_operation(self, operation_id: str):
        """Unregister a completed operation."""
        
        with self._lock:
            self._active_operations.pop(operation_id, None)
    
    def _handle_timeout(
        self,
        operation_id: str,
        reason: TimeoutReason,
        strategy: TimeoutStrategy,
        future
    ) -> TimeoutEvent:
        """Handle timeout based on configured strategy."""
        
        start_time = time.time()
        partial_result = None
        recovery_attempted = False
        recovery_successful = False
        
        try:
            if strategy == TimeoutStrategy.IMMEDIATE_STOP:
                # Cancel immediately
                future.cancel()
                
            elif strategy == TimeoutStrategy.GRACEFUL_FINISH:
                # Allow brief additional time for graceful completion
                recovery_attempted = True
                try:
                    partial_result = future.result(timeout=self.config.quick_recovery_timeout)
                    recovery_successful = True
                except FutureTimeoutError:
                    future.cancel()
                    
            elif strategy == TimeoutStrategy.BEST_EFFORT:
                # Try to get any partial result
                recovery_attempted = True
                partial_result = self._extract_partial_result(operation_id)
                if partial_result is not None:
                    recovery_successful = True
                future.cancel()
                
            elif strategy == TimeoutStrategy.INCREMENTAL_EXTEND:
                # Try extending timeout
                recovery_attempted = True
                extension_time = self.config.soft_timeout_seconds * self.config.extension_factor
                
                if self.request_timeout_extension(operation_id, extension_time, "Automatic extension"):
                    try:
                        partial_result = future.result(timeout=extension_time)
                        recovery_successful = True
                    except FutureTimeoutError:
                        future.cancel()
                        partial_result = self._extract_partial_result(operation_id)
                else:
                    future.cancel()
                    
            elif strategy == TimeoutStrategy.ADAPTIVE_SWITCH:
                # Switch to simpler strategy and continue
                recovery_attempted = True
                # This would require integration with reasoning engine
                # For now, just extract partial result
                partial_result = self._extract_partial_result(operation_id)
                if partial_result is not None:
                    recovery_successful = True
                future.cancel()
                
        except Exception as e:
            warnings.warn(f"Error during timeout handling: {str(e)}")
            future.cancel()
        
        return TimeoutEvent(
            timestamp=time.time(),
            reason=reason,
            strategy_used=strategy,
            partial_result=partial_result,
            steps_completed=self._get_steps_completed(operation_id),
            time_elapsed=time.time() - start_time,
            recovery_attempted=recovery_attempted,
            recovery_successful=recovery_successful
        )
    
    def _attempt_graceful_finish(self, operation_id: str, trace: ReasoningTrace) -> TimeoutEvent:
        """Attempt to gracefully finish reasoning session."""
        
        # Create a conclusion step with current best answer
        if trace.steps:
            best_step = max(trace.steps, key=lambda s: s.confidence)
            
            # Create final conclusion
            final_answer = f"Based on analysis so far: {best_step.content}"
            partial_result = ReasoningTrace(
                trace_id=trace.trace_id,
                query=trace.query,
                strategy=trace.strategy,
                start_time=trace.start_time,
                end_time=time.time(),
                steps=trace.steps,
                step_index=trace.step_index,
                final_answer=final_answer,
                confidence=best_step.confidence * 0.8,  # Reduce confidence for early termination
                total_reward=trace.total_reward,
                total_steps=trace.total_steps,
                successful_steps=trace.successful_steps,
                backtrack_count=trace.backtrack_count
            )
        else:
            partial_result = None
        
        return TimeoutEvent(
            timestamp=time.time(),
            reason=TimeoutReason.SOFT_TIMEOUT,
            strategy_used=TimeoutStrategy.GRACEFUL_FINISH,
            partial_result=partial_result,
            steps_completed=len(trace.steps),
            time_elapsed=time.time() - trace.start_time,
            recovery_attempted=True,
            recovery_successful=partial_result is not None
        )
    
    def _create_timeout_event(
        self,
        reason: TimeoutReason,
        strategy: TimeoutStrategy,
        operation_id: str,
        trace: ReasoningTrace
    ) -> TimeoutEvent:
        """Create a timeout event."""
        
        return TimeoutEvent(
            timestamp=time.time(),
            reason=reason,
            strategy_used=strategy,
            partial_result=trace if self.config.enable_partial_results else None,
            steps_completed=len(trace.steps),
            time_elapsed=time.time() - trace.start_time,
            recovery_attempted=False,
            recovery_successful=False
        )
    
    def _extract_partial_result(self, operation_id: str) -> Optional[Any]:
        """Extract partial result from operation."""
        
        with self._lock:
            if operation_id in self._active_operations:
                return self._active_operations[operation_id].get("partial_result")
        
        return None
    
    def _get_steps_completed(self, operation_id: str) -> int:
        """Get number of steps completed for operation."""
        
        with self._lock:
            if operation_id in self._active_operations:
                return self._active_operations[operation_id].get("steps_completed", 0)
        
        return 0
    
    def _check_resource_exhaustion(self) -> bool:
        """Check if system resources are exhausted."""
        
        try:
            import psutil
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:  # 90% memory usage
                return True
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.config.max_cpu_percent:
                return True
            
            return False
            
        except ImportError:
            # psutil not available, skip resource monitoring
            return False
        except Exception as e:
            warnings.warn(f"Error checking resources: {str(e)}")
            return False
    
    def _start_monitoring(self):
        """Start background monitoring thread."""
        
        def monitor_loop():
            while not self._stop_monitoring.is_set():
                try:
                    current_time = time.time()
                    
                    with self._lock:
                        operations_to_check = list(self._active_operations.items())
                    
                    for op_id, operation in operations_to_check:
                        elapsed = current_time - operation["start_time"]
                        
                        # Check for step timeout
                        if elapsed > self.config.step_timeout_seconds:
                            # This would require integration with reasoning engine
                            # For now, just mark as potentially problematic
                            operation["potential_timeout"] = True
                        
                        # Check for cancellation
                        if operation.get("cancelled", False):
                            self._unregister_operation(op_id)
                    
                    # Sleep before next check
                    time.sleep(5.0)
                    
                except Exception as e:
                    warnings.warn(f"Error in monitoring loop: {str(e)}")
                    time.sleep(5.0)
        
        self._monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitoring_thread.start()
    
    def _stop_monitoring_thread(self):
        """Stop the monitoring thread."""
        
        if self._monitoring_thread:
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5.0)
    
    def __del__(self):
        """Cleanup when handler is destroyed."""
        self._stop_monitoring_thread()


@contextmanager
def timeout_protection(
    timeout_seconds: float,
    strategy: TimeoutStrategy = TimeoutStrategy.GRACEFUL_FINISH,
    operation_id: Optional[str] = None
):
    """Context manager for timeout protection."""
    
    handler = TimeoutHandler(TimeoutConfig(
        soft_timeout_seconds=timeout_seconds,
        hard_timeout_seconds=timeout_seconds * 1.2,
        default_strategy=strategy
    ))
    
    operation_id = operation_id or f"ctx_{int(time.time() * 1000)}"
    start_time = time.time()
    
    handler._register_operation(operation_id, "context_manager", start_time, timeout_seconds)
    
    try:
        yield handler
    finally:
        handler._unregister_operation(operation_id)


class AdaptiveTimeoutManager:
    """Manages adaptive timeout adjustments based on performance history."""
    
    def __init__(self, base_config: TimeoutConfig):
        self.base_config = base_config
        self.performance_history: List[Dict[str, Any]] = []
        self.strategy_performance: Dict[ReasoningStrategy, Dict[str, float]] = {}
        
    def get_adaptive_timeout(
        self,
        strategy: ReasoningStrategy,
        query_complexity: str,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Get adaptive timeout based on strategy and complexity."""
        
        base_timeout = self.base_config.soft_timeout_seconds
        
        # Strategy-based adjustments
        strategy_multipliers = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: 1.0,
            ReasoningStrategy.TREE_OF_THOUGHT: 1.5,
            ReasoningStrategy.GRAPH_OF_THOUGHT: 2.0,
            ReasoningStrategy.SELF_CONSISTENCY: 1.3,
            ReasoningStrategy.METACOGNITIVE: 1.8
        }
        
        multiplier = strategy_multipliers.get(strategy, 1.0)
        
        # Complexity adjustments
        complexity_adjustments = {
            "simple": 0.5,
            "moderate": 1.0,
            "complex": 1.5,
            "very_complex": 2.0
        }
        
        complexity_factor = complexity_adjustments.get(query_complexity, 1.0)
        
        # Historical performance adjustments
        if strategy in self.strategy_performance:
            perf = self.strategy_performance[strategy]
            avg_time = perf.get("avg_time", base_timeout)
            success_rate = perf.get("success_rate", 0.5)
            
            # Adjust based on historical success and timing
            if success_rate > 0.8:
                # High success rate, can be more aggressive with timeouts
                historical_factor = 0.9
            elif success_rate < 0.5:
                # Low success rate, give more time
                historical_factor = 1.3
            else:
                historical_factor = 1.0
            
            # Consider average time
            if avg_time > 0:
                time_factor = avg_time / base_timeout
                historical_factor *= min(2.0, max(0.5, time_factor))
        else:
            historical_factor = 1.0
        
        adaptive_timeout = base_timeout * multiplier * complexity_factor * historical_factor
        
        # Ensure reasonable bounds
        min_timeout = 30.0   # 30 seconds minimum
        max_timeout = 600.0  # 10 minutes maximum
        
        return max(min_timeout, min(adaptive_timeout, max_timeout))
    
    def update_performance(
        self,
        strategy: ReasoningStrategy,
        actual_time: float,
        success: bool,
        timeout_occurred: bool
    ):
        """Update performance history for adaptive adjustments."""
        
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                "total_runs": 0,
                "total_time": 0.0,
                "successes": 0,
                "timeouts": 0
            }
        
        perf = self.strategy_performance[strategy]
        perf["total_runs"] += 1
        perf["total_time"] += actual_time
        
        if success:
            perf["successes"] += 1
        
        if timeout_occurred:
            perf["timeouts"] += 1
        
        # Calculate derived metrics
        perf["avg_time"] = perf["total_time"] / perf["total_runs"]
        perf["success_rate"] = perf["successes"] / perf["total_runs"]
        perf["timeout_rate"] = perf["timeouts"] / perf["total_runs"]
        
        # Store in history
        self.performance_history.append({
            "timestamp": time.time(),
            "strategy": strategy.value,
            "actual_time": actual_time,
            "success": success,
            "timeout_occurred": timeout_occurred
        })
        
        # Keep history bounded
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]