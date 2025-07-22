"""A/B testing framework for LoRA adapters in SEAL architecture.

This module provides comprehensive A/B testing capabilities for comparing
different LoRA configurations and evaluating their performance.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
import time
import json
import numpy as np
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlite3
import scipy.stats

from .lora_adapter import LoRALinear, LoRAEmbedding, LoRAConfig
from .memory_efficient_lora import MemoryEfficientLoRA
from .performance_analyzer import PerformanceSnapshot


@dataclass
class ABTestConfig:
    """Configuration for A/B testing experiments."""
    # Test parameters
    test_name: str
    description: str = ""
    
    # Sample allocation
    traffic_split: Dict[str, float] = field(default_factory=lambda: {"A": 0.5, "B": 0.5})
    min_samples_per_variant: int = 100
    max_samples_per_variant: int = 10000
    
    # Statistical settings
    confidence_level: float = 0.95
    minimum_detectable_effect: float = 0.05  # 5% minimum effect size
    power: float = 0.8  # Statistical power
    
    # Success metrics
    primary_metric: str = "accuracy"
    secondary_metrics: List[str] = field(default_factory=lambda: ["latency", "memory_usage"])
    
    # Test duration
    max_duration_hours: float = 24.0
    early_stopping_enabled: bool = True
    early_stopping_threshold: float = 0.01  # p-value threshold
    
    # Quality controls
    enable_guardrails: bool = True
    max_acceptable_degradation: float = 0.1  # 10% max degradation
    
    # Data collection
    sample_data_for_analysis: bool = True
    max_sample_storage: int = 1000


@dataclass
class ABTestVariant:
    """Configuration for a single test variant."""
    variant_id: str
    name: str
    description: str = ""
    
    # LoRA configuration
    lora_config: Optional[LoRAConfig] = None
    adapter_weights: Dict[str, Union[LoRALinear, LoRAEmbedding]] = field(default_factory=dict)
    
    # Model configuration
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Expected allocation
    expected_traffic: float = 0.5
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class ABTestSample:
    """A single test sample/observation."""
    sample_id: str
    variant_id: str
    timestamp: float
    
    # Input/output data
    input_data: Any = None
    output_data: Any = None
    
    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Context
    user_context: Dict[str, Any] = field(default_factory=dict)
    model_context: Dict[str, Any] = field(default_factory=dict)
    
    # Quality indicators
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ABTestResult:
    """Results of an A/B test."""
    test_name: str
    
    # Test summary
    start_time: float
    end_time: float
    duration_hours: float
    total_samples: int
    
    # Variant results
    variant_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Statistical analysis
    primary_metric_results: Dict[str, Any] = field(default_factory=dict)
    secondary_metric_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Decision
    winning_variant: Optional[str] = None
    confidence: float = 0.0
    statistical_significance: bool = False
    
    # Recommendations
    recommendation: str = "no_decision"  # "adopt_variant", "keep_control", "extend_test"
    reasoning: str = ""
    
    # Quality metrics
    data_quality_score: float = 1.0
    test_validity: bool = True
    
    # Additional insights
    insights: List[str] = field(default_factory=list)


class ABTestManager:
    """Manages A/B tests for LoRA adapters."""
    
    def __init__(self, results_dir: Path = Path("ab_test_results")):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Database for test data
        self.db_path = self.results_dir / "ab_tests.db"
        self._init_database()
        
        # Active tests
        self.active_tests: Dict[str, 'ABTest'] = {}
        
        # Thread pool for concurrent testing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Traffic router
        self.traffic_router = TrafficRouter()
    
    def _init_database(self):
        """Initialize SQLite database for test data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tests table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ab_tests (
                    test_name TEXT PRIMARY KEY,
                    config_json TEXT,
                    start_time REAL,
                    end_time REAL,
                    status TEXT,
                    result_json TEXT
                )
            """)
            
            # Variants table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_variants (
                    test_name TEXT,
                    variant_id TEXT,
                    variant_config TEXT,
                    PRIMARY KEY (test_name, variant_id),
                    FOREIGN KEY (test_name) REFERENCES ab_tests (test_name)
                )
            """)
            
            # Samples table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_samples (
                    sample_id TEXT PRIMARY KEY,
                    test_name TEXT,
                    variant_id TEXT,
                    timestamp REAL,
                    metrics_json TEXT,
                    success INTEGER,
                    FOREIGN KEY (test_name) REFERENCES ab_tests (test_name)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_timestamp ON test_samples(test_name, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_variant_timestamp ON test_samples(variant_id, timestamp)")
            
            conn.commit()
    
    def create_test(
        self,
        config: ABTestConfig,
        variants: List[ABTestVariant]
    ) -> 'ABTest':
        """Create a new A/B test."""
        if config.test_name in self.active_tests:
            raise ValueError(f"Test {config.test_name} already exists")
        
        # Validate traffic splits
        total_traffic = sum(config.traffic_split.values())
        if abs(total_traffic - 1.0) > 0.01:
            raise ValueError(f"Traffic split must sum to 1.0, got {total_traffic}")
        
        # Create test instance
        test = ABTest(config, variants, self.db_path)
        self.active_tests[config.test_name] = test
        
        # Save to database
        self._save_test_config(config, variants)
        
        return test
    
    def get_test(self, test_name: str) -> Optional['ABTest']:
        """Get an active test by name."""
        return self.active_tests.get(test_name)
    
    def list_tests(self, status: Optional[str] = None) -> List[str]:
        """List available tests."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if status:
                cursor.execute("SELECT test_name FROM ab_tests WHERE status = ?", (status,))
            else:
                cursor.execute("SELECT test_name FROM ab_tests")
            
            return [row[0] for row in cursor.fetchall()]
    
    def run_inference_test(
        self,
        test_name: str,
        model: nn.Module,
        test_inputs: List[Any],
        evaluation_fn: Callable[[Any, Any], Dict[str, float]]
    ) -> ABTestResult:
        """Run an inference-based A/B test."""
        test = self.get_test(test_name)
        if not test:
            raise ValueError(f"Test {test_name} not found")
        
        # Run test on all inputs
        for input_data in test_inputs:
            # Route traffic to variant
            variant_id = self.traffic_router.route_traffic(
                test.config.traffic_split, input_data
            )
            
            # Apply variant configuration
            variant = test.get_variant(variant_id)
            self._apply_variant_to_model(model, variant)
            
            # Run inference
            try:
                with torch.no_grad():
                    output = model(input_data)
                
                # Evaluate metrics
                metrics = evaluation_fn(input_data, output)
                
                # Record sample
                test.record_sample(variant_id, input_data, output, metrics)
                
            except Exception as e:
                # Record failed sample
                test.record_sample(variant_id, input_data, None, {}, success=False, error=str(e))
        
        # Analyze results
        return test.analyze_results()
    
    def run_training_test(
        self,
        test_name: str,
        base_model: nn.Module,
        training_data: Any,
        training_fn: Callable[[nn.Module, Any], Dict[str, float]]
    ) -> ABTestResult:
        """Run a training-based A/B test."""
        test = self.get_test(test_name)
        if not test:
            raise ValueError(f"Test {test_name} not found")
        
        # Run training for each variant
        variant_results = {}
        
        for variant in test.variants:
            # Create model copy for this variant
            model_copy = self._create_model_copy(base_model)
            self._apply_variant_to_model(model_copy, variant)
            
            # Run training
            training_metrics = training_fn(model_copy, training_data)
            
            # Record results
            variant_results[variant.variant_id] = training_metrics
            
            # Record as samples for analysis
            for metric_name, value in training_metrics.items():
                test.record_sample(
                    variant.variant_id,
                    input_data="training",
                    output_data=None,
                    metrics={metric_name: value}
                )
        
        return test.analyze_results()
    
    def _apply_variant_to_model(self, model: nn.Module, variant: ABTestVariant):
        """Apply variant configuration to model."""
        # Apply LoRA adapters
        for layer_name, adapter in variant.adapter_weights.items():
            # Find the target layer in the model
            target_layer = self._find_layer_by_name(model, layer_name)
            if target_layer and hasattr(target_layer, 'set_adapter'):
                target_layer.set_adapter(adapter)
        
        # Apply model parameters
        if variant.model_params:
            for param_name, value in variant.model_params.items():
                if hasattr(model, param_name):
                    setattr(model, param_name, value)
    
    def _find_layer_by_name(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Find a layer in the model by name."""
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        return None
    
    def _create_model_copy(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model."""
        # Create new instance
        model_copy = type(model)()
        
        # Copy state dict
        model_copy.load_state_dict(model.state_dict())
        
        return model_copy
    
    def _save_test_config(self, config: ABTestConfig, variants: List[ABTestVariant]):
        """Save test configuration to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Save test config
            cursor.execute("""
                INSERT OR REPLACE INTO ab_tests 
                (test_name, config_json, start_time, status)
                VALUES (?, ?, ?, ?)
            """, (
                config.test_name,
                json.dumps(asdict(config)),
                time.time(),
                "created"
            ))
            
            # Save variants
            for variant in variants:
                cursor.execute("""
                    INSERT OR REPLACE INTO test_variants
                    (test_name, variant_id, variant_config)
                    VALUES (?, ?, ?)
                """, (
                    config.test_name,
                    variant.variant_id,
                    json.dumps(asdict(variant))
                ))
            
            conn.commit()


class ABTest:
    """Individual A/B test instance."""
    
    def __init__(self, config: ABTestConfig, variants: List[ABTestVariant], db_path: Path):
        self.config = config
        self.variants = {v.variant_id: v for v in variants}
        self.db_path = db_path
        
        # Test state
        self.start_time = time.time()
        self.status = "running"
        self.samples = defaultdict(list)
        
        # Statistics tracking
        self.sample_counts = defaultdict(int)
        self.metric_accumulators = defaultdict(lambda: defaultdict(list))
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def get_variant(self, variant_id: str) -> ABTestVariant:
        """Get variant by ID."""
        if variant_id not in self.variants:
            raise ValueError(f"Variant {variant_id} not found")
        return self.variants[variant_id]
    
    def record_sample(
        self,
        variant_id: str,
        input_data: Any,
        output_data: Any,
        metrics: Dict[str, float],
        success: bool = True,
        error: Optional[str] = None
    ):
        """Record a test sample."""
        with self.lock:
            sample_id = f"{self.config.test_name}_{variant_id}_{int(time.time() * 1000)}"
            
            sample = ABTestSample(
                sample_id=sample_id,
                variant_id=variant_id,
                timestamp=time.time(),
                input_data=input_data if self.config.sample_data_for_analysis else None,
                output_data=output_data if self.config.sample_data_for_analysis else None,
                metrics=metrics,
                success=success,
                error_message=error
            )
            
            # Store sample
            self.samples[variant_id].append(sample)
            self.sample_counts[variant_id] += 1
            
            # Update metric accumulator
            for metric_name, value in metrics.items():
                self.metric_accumulators[variant_id][metric_name].append(value)
            
            # Save to database
            self._save_sample_to_db(sample)
            
            # Check for early stopping
            if self.config.early_stopping_enabled:
                self._check_early_stopping()
    
    def analyze_results(self) -> ABTestResult:
        """Analyze test results and generate report."""
        end_time = time.time()
        
        # Calculate variant results
        variant_results = {}
        for variant_id in self.variants.keys():
            variant_results[variant_id] = self._analyze_variant(variant_id)
        
        # Statistical analysis of primary metric
        primary_analysis = self._analyze_primary_metric()
        
        # Secondary metrics analysis
        secondary_analysis = {}
        for metric in self.config.secondary_metrics:
            secondary_analysis[metric] = self._analyze_secondary_metric(metric)
        
        # Determine winner and recommendation
        winning_variant, confidence, recommendation, reasoning = self._make_decision(
            primary_analysis, variant_results
        )
        
        result = ABTestResult(
            test_name=self.config.test_name,
            start_time=self.start_time,
            end_time=end_time,
            duration_hours=(end_time - self.start_time) / 3600,
            total_samples=sum(self.sample_counts.values()),
            variant_results=variant_results,
            primary_metric_results=primary_analysis,
            secondary_metric_results=secondary_analysis,
            winning_variant=winning_variant,
            confidence=confidence,
            statistical_significance=primary_analysis.get('significant', False),
            recommendation=recommendation,
            reasoning=reasoning,
            data_quality_score=self._calculate_data_quality(),
            test_validity=self._validate_test_results(),
            insights=self._generate_insights(variant_results, primary_analysis)
        )
        
        # Save results
        self._save_results_to_db(result)
        
        return result
    
    def _analyze_variant(self, variant_id: str) -> Dict[str, Any]:
        """Analyze results for a single variant."""
        samples = self.samples[variant_id]
        
        if not samples:
            return {"sample_count": 0, "success_rate": 0.0, "metrics": {}}
        
        # Basic statistics
        sample_count = len(samples)
        successful_samples = [s for s in samples if s.success]
        success_rate = len(successful_samples) / sample_count
        
        # Metric statistics
        metric_stats = {}
        for metric_name in self.config.secondary_metrics + [self.config.primary_metric]:
            values = [s.metrics.get(metric_name, 0) for s in successful_samples 
                     if metric_name in s.metrics]
            
            if values:
                metric_stats[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "median": np.median(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values)
                }
        
        return {
            "sample_count": sample_count,
            "success_rate": success_rate,
            "metrics": metric_stats
        }
    
    def _analyze_primary_metric(self) -> Dict[str, Any]:
        """Perform statistical analysis of the primary metric."""
        metric_name = self.config.primary_metric
        variant_ids = list(self.variants.keys())
        
        if len(variant_ids) != 2:
            # For now, only support two-variant tests
            return {"error": "Only two-variant tests supported for statistical analysis"}
        
        # Get data for both variants
        variant_a_data = self.metric_accumulators[variant_ids[0]][metric_name]
        variant_b_data = self.metric_accumulators[variant_ids[1]][metric_name]
        
        if not variant_a_data or not variant_b_data:
            return {"error": "Insufficient data for statistical analysis"}
        
        # Perform t-test
        statistic, p_value = scipy.stats.ttest_ind(variant_a_data, variant_b_data)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(variant_a_data) - 1) * np.var(variant_a_data, ddof=1) +
             (len(variant_b_data) - 1) * np.var(variant_b_data, ddof=1)) /
            (len(variant_a_data) + len(variant_b_data) - 2)
        )
        
        effect_size = (np.mean(variant_a_data) - np.mean(variant_b_data)) / pooled_std
        
        # Confidence interval
        alpha = 1 - self.config.confidence_level
        
        return {
            "metric": metric_name,
            "variant_a_mean": np.mean(variant_a_data),
            "variant_b_mean": np.mean(variant_b_data),
            "difference": np.mean(variant_b_data) - np.mean(variant_a_data),
            "relative_improvement": (np.mean(variant_b_data) - np.mean(variant_a_data)) / np.mean(variant_a_data),
            "t_statistic": statistic,
            "p_value": p_value,
            "effect_size": effect_size,
            "significant": p_value < alpha,
            "confidence_level": self.config.confidence_level
        }
    
    def _analyze_secondary_metric(self, metric_name: str) -> Dict[str, Any]:
        """Analyze a secondary metric."""
        variant_ids = list(self.variants.keys())
        
        if len(variant_ids) != 2:
            return {"error": "Only two-variant tests supported"}
        
        variant_a_data = self.metric_accumulators[variant_ids[0]][metric_name]
        variant_b_data = self.metric_accumulators[variant_ids[1]][metric_name]
        
        if not variant_a_data or not variant_b_data:
            return {"error": "Insufficient data"}
        
        # Simple comparison for secondary metrics
        return {
            "metric": metric_name,
            "variant_a_mean": np.mean(variant_a_data),
            "variant_b_mean": np.mean(variant_b_data),
            "difference": np.mean(variant_b_data) - np.mean(variant_a_data),
            "relative_change": (np.mean(variant_b_data) - np.mean(variant_a_data)) / np.mean(variant_a_data)
        }
    
    def _make_decision(
        self,
        primary_analysis: Dict[str, Any],
        variant_results: Dict[str, Any]
    ) -> Tuple[Optional[str], float, str, str]:
        """Make a decision based on test results."""
        if "error" in primary_analysis:
            return None, 0.0, "no_decision", "Insufficient data for decision"
        
        # Check statistical significance
        if not primary_analysis.get("significant", False):
            return None, 0.0, "extend_test", "No statistically significant difference found"
        
        # Check minimum detectable effect
        relative_improvement = abs(primary_analysis.get("relative_improvement", 0))
        if relative_improvement < self.config.minimum_detectable_effect:
            return None, 0.0, "keep_control", "Effect size below minimum detectable threshold"
        
        # Check guardrails
        if self.config.enable_guardrails:
            improvement = primary_analysis.get("relative_improvement", 0)
            if improvement < -self.config.max_acceptable_degradation:
                return None, 0.0, "keep_control", "Variant shows unacceptable degradation"
        
        # Determine winner
        variant_ids = list(self.variants.keys())
        if primary_analysis.get("relative_improvement", 0) > 0:
            winner = variant_ids[1]  # Variant B
            reasoning = f"Variant B shows {relative_improvement:.2%} improvement"
        else:
            winner = variant_ids[0]  # Variant A
            reasoning = f"Variant A shows {relative_improvement:.2%} improvement"
        
        confidence = 1 - primary_analysis.get("p_value", 1.0)
        
        return winner, confidence, "adopt_variant", reasoning
    
    def _calculate_data_quality(self) -> float:
        """Calculate overall data quality score."""
        total_samples = sum(self.sample_counts.values())
        if total_samples == 0:
            return 0.0
        
        # Success rate across all variants
        successful_samples = sum(
            len([s for s in samples if s.success])
            for samples in self.samples.values()
        )
        
        success_rate = successful_samples / total_samples
        
        # Sample size adequacy
        min_samples = min(self.sample_counts.values()) if self.sample_counts else 0
        sample_adequacy = min(min_samples / self.config.min_samples_per_variant, 1.0)
        
        # Balance between variants
        expected_counts = {
            variant_id: total_samples * self.config.traffic_split.get(variant_id, 0.5)
            for variant_id in self.variants.keys()
        }
        
        balance_score = 1.0
        for variant_id, expected in expected_counts.items():
            if expected > 0:
                actual = self.sample_counts[variant_id]
                deviation = abs(actual - expected) / expected
                balance_score *= max(0, 1 - deviation)
        
        # Combined score
        quality_score = 0.4 * success_rate + 0.3 * sample_adequacy + 0.3 * balance_score
        
        return quality_score
    
    def _validate_test_results(self) -> bool:
        """Validate test results for common issues."""
        # Check minimum sample sizes
        for variant_id, count in self.sample_counts.items():
            if count < self.config.min_samples_per_variant:
                return False
        
        # Check for extreme imbalances
        total_samples = sum(self.sample_counts.values())
        for variant_id, count in self.sample_counts.items():
            expected_ratio = self.config.traffic_split.get(variant_id, 0.5)
            actual_ratio = count / total_samples if total_samples > 0 else 0
            
            if abs(actual_ratio - expected_ratio) > 0.2:  # 20% deviation threshold
                return False
        
        return True
    
    def _generate_insights(
        self,
        variant_results: Dict[str, Any],
        primary_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate insights from test results."""
        insights = []
        
        # Sample size insights
        total_samples = sum(self.sample_counts.values())
        insights.append(f"Collected {total_samples} total samples across {len(self.variants)} variants")
        
        # Performance insights
        if "relative_improvement" in primary_analysis:
            improvement = primary_analysis["relative_improvement"]
            if abs(improvement) > 0.1:  # 10% threshold
                direction = "improvement" if improvement > 0 else "degradation"
                insights.append(f"Observed {abs(improvement):.2%} {direction} in primary metric")
        
        # Success rate insights
        success_rates = {
            variant_id: results.get("success_rate", 0)
            for variant_id, results in variant_results.items()
        }
        
        min_success = min(success_rates.values())
        max_success = max(success_rates.values())
        
        if max_success - min_success > 0.1:  # 10% difference
            insights.append(f"Success rates vary significantly between variants ({min_success:.1%} - {max_success:.1%})")
        
        # Statistical power insight
        if "p_value" in primary_analysis:
            p_value = primary_analysis["p_value"]
            if 0.05 < p_value < 0.1:
                insights.append("Results are marginally significant - consider collecting more data")
        
        return insights
    
    def _check_early_stopping(self):
        """Check if test should be stopped early."""
        if not self.config.early_stopping_enabled:
            return
        
        # Check minimum samples
        if min(self.sample_counts.values()) < self.config.min_samples_per_variant:
            return
        
        # Perform interim analysis
        primary_analysis = self._analyze_primary_metric()
        
        if "p_value" in primary_analysis:
            p_value = primary_analysis["p_value"]
            
            # Early stopping for significance
            if p_value < self.config.early_stopping_threshold:
                self.status = "stopped_early_significant"
                print(f"Test {self.config.test_name} stopped early due to statistical significance")
            
            # Early stopping for futility (very high p-value)
            elif p_value > 0.8 and sum(self.sample_counts.values()) > 500:
                self.status = "stopped_early_futile"
                print(f"Test {self.config.test_name} stopped early due to futility")
    
    def _save_sample_to_db(self, sample: ABTestSample):
        """Save sample to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO test_samples 
                (sample_id, test_name, variant_id, timestamp, metrics_json, success)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                sample.sample_id,
                self.config.test_name,
                sample.variant_id,
                sample.timestamp,
                json.dumps(sample.metrics),
                1 if sample.success else 0
            ))
            
            conn.commit()
    
    def _save_results_to_db(self, result: ABTestResult):
        """Save test results to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE ab_tests 
                SET end_time = ?, status = ?, result_json = ?
                WHERE test_name = ?
            """, (
                result.end_time,
                "completed",
                json.dumps(asdict(result)),
                result.test_name
            ))
            
            conn.commit()


class TrafficRouter:
    """Routes traffic to different test variants."""
    
    def __init__(self):
        self.routing_cache = {}
    
    def route_traffic(
        self,
        traffic_split: Dict[str, float],
        input_data: Any,
        sticky_routing: bool = False
    ) -> str:
        """Route traffic to a variant based on split configuration."""
        if sticky_routing and input_data in self.routing_cache:
            return self.routing_cache[input_data]
        
        # Generate deterministic random number based on input
        import hashlib
        
        input_hash = hashlib.md5(str(input_data).encode()).hexdigest()
        random_value = int(input_hash[:8], 16) / 0xffffffff
        
        # Route based on cumulative distribution
        cumulative = 0.0
        for variant_id, probability in traffic_split.items():
            cumulative += probability
            if random_value <= cumulative:
                if sticky_routing:
                    self.routing_cache[input_data] = variant_id
                return variant_id
        
        # Fallback to last variant
        return list(traffic_split.keys())[-1]


# Utility functions for A/B testing

def create_simple_ab_test(
    test_name: str,
    control_adapters: Dict[str, Union[LoRALinear, LoRAEmbedding]],
    treatment_adapters: Dict[str, Union[LoRALinear, LoRAEmbedding]],
    primary_metric: str = "accuracy",
    traffic_split: Tuple[float, float] = (0.5, 0.5)
) -> Tuple[ABTestConfig, List[ABTestVariant]]:
    """Create a simple A/B test configuration."""
    config = ABTestConfig(
        test_name=test_name,
        traffic_split={"control": traffic_split[0], "treatment": traffic_split[1]},
        primary_metric=primary_metric
    )
    
    variants = [
        ABTestVariant(
            variant_id="control",
            name="Control",
            adapter_weights=control_adapters,
            expected_traffic=traffic_split[0]
        ),
        ABTestVariant(
            variant_id="treatment",
            name="Treatment",
            adapter_weights=treatment_adapters,
            expected_traffic=traffic_split[1]
        )
    ]
    
    return config, variants


def analyze_test_power(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    baseline_std: float = 1.0
) -> int:
    """Calculate required sample size for desired statistical power."""
    # Simplified power analysis
    from scipy import stats
    
    # Effect size in standard deviations
    d = effect_size / baseline_std
    
    # Critical value
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    
    # Sample size per group
    n = 2 * ((z_alpha + z_beta) / d) ** 2
    
    return int(np.ceil(n))


def visualize_test_results(result: ABTestResult, save_path: Optional[Path] = None):
    """Create visualizations for A/B test results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Primary metric comparison
    ax1 = axes[0, 0]
    variants = list(result.variant_results.keys())
    primary_metric = result.primary_metric_results.get("metric", "primary_metric")
    
    values = [
        result.variant_results[v]["metrics"].get(primary_metric, {}).get("mean", 0)
        for v in variants
    ]
    
    ax1.bar(variants, values)
    ax1.set_title(f"Primary Metric: {primary_metric}")
    ax1.set_ylabel("Mean Value")
    
    # Sample counts
    ax2 = axes[0, 1]
    sample_counts = [result.variant_results[v]["sample_count"] for v in variants]
    ax2.bar(variants, sample_counts)
    ax2.set_title("Sample Counts")
    ax2.set_ylabel("Number of Samples")
    
    # Success rates
    ax3 = axes[1, 0]
    success_rates = [result.variant_results[v]["success_rate"] for v in variants]
    ax3.bar(variants, success_rates)
    ax3.set_title("Success Rates")
    ax3.set_ylabel("Success Rate")
    ax3.set_ylim(0, 1)
    
    # Effect size visualization
    ax4 = axes[1, 1]
    if "relative_improvement" in result.primary_metric_results:
        improvement = result.primary_metric_results["relative_improvement"]
        ax4.bar(["Effect Size"], [improvement])
        ax4.set_title("Relative Improvement")
        ax4.set_ylabel("Relative Change")
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()