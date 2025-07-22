"""Model switching logic for dynamic model selection."""

import time
import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import psutil

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Types of tasks for model selection."""
    SIMPLE_QA = "simple_qa"
    COMPLEX_REASONING = "complex_reasoning"
    CODE_GENERATION = "code_generation"
    CREATIVE_WRITING = "creative_writing"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    CONSCIOUSNESS_TASK = "consciousness"

@dataclass
class ModelProfile:
    """Profile for each model's capabilities."""
    name: str
    size_gb: float
    speed_tokens_per_sec: float
    strengths: List[TaskType]
    quality_score: float  # 0-1 scale
    memory_required_gb: float  # Including KV cache

class ModelSwitcher:
    """Intelligent model switching based on task requirements."""
    
    def __init__(self):
        # Model profiles based on benchmarks
        self.model_profiles = {
            "llama3.1:8b": ModelProfile(
                name="llama3.1:8b",
                size_gb=4.9,
                speed_tokens_per_sec=25.0,  # Observed average
                strengths=[TaskType.SIMPLE_QA, TaskType.CODE_GENERATION],
                quality_score=0.7,
                memory_required_gb=8.0
            ),
            "gemma3:27b": ModelProfile(
                name="gemma3:27b",
                size_gb=17.4,
                speed_tokens_per_sec=5.5,  # Observed average
                strengths=[TaskType.COMPLEX_REASONING, TaskType.CREATIVE_WRITING],
                quality_score=0.85,
                memory_required_gb=25.0
            ),
            "Qwen3:32b": ModelProfile(
                name="Qwen3:32b",
                size_gb=20.2,
                speed_tokens_per_sec=3.0,  # Estimated from timeout
                strengths=[TaskType.CONSCIOUSNESS_TASK, TaskType.COMPLEX_REASONING, 
                          TaskType.TRANSLATION],
                quality_score=0.95,
                memory_required_gb=30.0
            )
        }
        
        self.current_model = None
        self.switch_history = []
        self.performance_cache = {}
        
    def get_available_memory(self) -> float:
        """Get available system memory in GB."""
        memory = psutil.virtual_memory()
        return memory.available / (1024 ** 3)
        
    def classify_task(self, prompt: str) -> TaskType:
        """Classify the task type based on the prompt."""
        prompt_lower = prompt.lower()
        
        # Simple heuristics for task classification
        if any(word in prompt_lower for word in ["code", "function", "program", "algorithm"]):
            return TaskType.CODE_GENERATION
        elif any(word in prompt_lower for word in ["translate", "translation", "in chinese", "in spanish"]):
            return TaskType.TRANSLATION
        elif any(word in prompt_lower for word in ["summarize", "summary", "brief"]):
            return TaskType.SUMMARIZATION
        elif any(word in prompt_lower for word in ["story", "poem", "creative", "imagine"]):
            return TaskType.CREATIVE_WRITING
        elif any(word in prompt_lower for word in ["consciousness", "self-aware", "sentient", "think about thinking"]):
            return TaskType.CONSCIOUSNESS_TASK
        elif any(word in prompt_lower for word in ["explain", "analyze", "compare", "evaluate"]):
            return TaskType.COMPLEX_REASONING
        else:
            return TaskType.SIMPLE_QA
            
    def score_model_for_task(self, model_profile: ModelProfile, task_type: TaskType,
                           required_speed: Optional[float] = None,
                           required_quality: Optional[float] = None) -> float:
        """Score a model's suitability for a task."""
        score = 0.0
        
        # Task fit score (0-0.4)
        if task_type in model_profile.strengths:
            score += 0.4
        else:
            score += 0.1
            
        # Quality score (0-0.3)
        quality_weight = 0.3
        if required_quality:
            if model_profile.quality_score >= required_quality:
                score += quality_weight
            else:
                score += quality_weight * (model_profile.quality_score / required_quality)
        else:
            score += quality_weight * model_profile.quality_score
            
        # Speed score (0-0.2)
        speed_weight = 0.2
        if required_speed:
            if model_profile.speed_tokens_per_sec >= required_speed:
                score += speed_weight
            else:
                score += speed_weight * (model_profile.speed_tokens_per_sec / required_speed)
        else:
            # Normalize speed score (assuming 50 tokens/s is ideal)
            score += speed_weight * min(model_profile.speed_tokens_per_sec / 50, 1.0)
            
        # Memory feasibility (0-0.1)
        available_memory = self.get_available_memory()
        if model_profile.memory_required_gb <= available_memory:
            score += 0.1
            
        return score
        
    def select_model(self, 
                    prompt: str,
                    task_type: Optional[TaskType] = None,
                    required_speed: Optional[float] = None,
                    required_quality: Optional[float] = None,
                    max_latency_ms: Optional[float] = None) -> str:
        """Select the best model for the given task."""
        
        # Auto-classify task if not provided
        if task_type is None:
            task_type = self.classify_task(prompt)
            
        logger.info(f"Task classified as: {task_type.value}")
        
        # Convert max latency to required speed if provided
        if max_latency_ms and not required_speed:
            # Assume average response of 100 tokens
            required_speed = 100 / (max_latency_ms / 1000)
            
        # Score all models
        model_scores = []
        for model_name, profile in self.model_profiles.items():
            score = self.score_model_for_task(
                profile, task_type, required_speed, required_quality
            )
            model_scores.append((model_name, score))
            logger.debug(f"{model_name} score: {score:.3f}")
            
        # Sort by score
        model_scores.sort(key=lambda x: x[1], reverse=True)
        selected_model = model_scores[0][0]
        
        # Record the switch
        if self.current_model != selected_model:
            self.switch_history.append({
                "from": self.current_model,
                "to": selected_model,
                "task_type": task_type.value,
                "timestamp": time.time()
            })
            logger.info(f"Switching from {self.current_model} to {selected_model}")
            self.current_model = selected_model
            
        return selected_model
        
    def get_model_recommendation(self, task_type: TaskType) -> List[Tuple[str, float]]:
        """Get model recommendations for a task type."""
        recommendations = []
        
        for model_name, profile in self.model_profiles.items():
            score = self.score_model_for_task(profile, task_type)
            recommendations.append((model_name, score))
            
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations
        
    def update_performance_metrics(self, model: str, task_type: TaskType,
                                 actual_tokens_per_sec: float, quality_rating: float):
        """Update model performance based on actual results."""
        key = f"{model}_{task_type.value}"
        
        if key not in self.performance_cache:
            self.performance_cache[key] = {
                "measurements": [],
                "avg_speed": 0,
                "avg_quality": 0
            }
            
        # Add new measurement
        self.performance_cache[key]["measurements"].append({
            "speed": actual_tokens_per_sec,
            "quality": quality_rating,
            "timestamp": time.time()
        })
        
        # Update averages (keep last 10 measurements)
        measurements = self.performance_cache[key]["measurements"][-10:]
        self.performance_cache[key]["avg_speed"] = sum(m["speed"] for m in measurements) / len(measurements)
        self.performance_cache[key]["avg_quality"] = sum(m["quality"] for m in measurements) / len(measurements)
        
        # Update model profile if significant difference
        if model in self.model_profiles:
            profile = self.model_profiles[model]
            
            # Update speed if >20% different
            if abs(profile.speed_tokens_per_sec - actual_tokens_per_sec) / profile.speed_tokens_per_sec > 0.2:
                old_speed = profile.speed_tokens_per_sec
                profile.speed_tokens_per_sec = 0.7 * profile.speed_tokens_per_sec + 0.3 * actual_tokens_per_sec
                logger.info(f"Updated {model} speed: {old_speed:.1f} -> {profile.speed_tokens_per_sec:.1f}")
                
    def get_switch_statistics(self) -> Dict:
        """Get statistics about model switching."""
        if not self.switch_history:
            return {"switches": 0}
            
        stats = {
            "switches": len(self.switch_history),
            "models_used": list(set(s["to"] for s in self.switch_history if s["to"])),
            "task_distribution": {}
        }
        
        # Count task types
        for switch in self.switch_history:
            task = switch["task_type"]
            stats["task_distribution"][task] = stats["task_distribution"].get(task, 0) + 1
            
        return stats

# Example usage functions
def create_model_switcher() -> ModelSwitcher:
    """Create a configured model switcher."""
    return ModelSwitcher()

def test_model_selection():
    """Test model selection logic."""
    switcher = create_model_switcher()
    
    test_cases = [
        ("What is 2+2?", TaskType.SIMPLE_QA),
        ("Write a Python function to implement quicksort", TaskType.CODE_GENERATION),
        ("Explain the philosophical implications of consciousness in AI", TaskType.CONSCIOUSNESS_TASK),
        ("Translate this to Chinese: Hello world", TaskType.TRANSLATION),
        ("Write a creative story about a robot", TaskType.CREATIVE_WRITING)
    ]
    
    print("=== Model Selection Test ===")
    for prompt, expected_type in test_cases:
        selected = switcher.select_model(prompt)
        actual_type = switcher.classify_task(prompt)
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"  Task type: {actual_type.value} (expected: {expected_type.value})")
        print(f"  Selected model: {selected}")
        
        # Get recommendations
        recommendations = switcher.get_model_recommendation(actual_type)
        print("  All scores:", {m: f"{s:.3f}" for m, s in recommendations})
        
    # Print statistics
    print(f"\n{switcher.get_switch_statistics()}")