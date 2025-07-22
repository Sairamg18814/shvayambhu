"""Benchmark token generation speeds for Ollama models."""

import time
import json
import statistics
from typing import Dict, List
import sys
sys.path.append('.')

from training.bootstrap.ollama_wrapper import OllamaWrapper, OllamaModel, GenerationConfig

class OllamaBenchmark:
    """Benchmark Ollama model performance."""
    
    def __init__(self):
        self.wrapper = OllamaWrapper()
        self.test_prompts = [
            "Explain quantum computing in simple terms.",
            "Write a Python function to sort a list.",
            "What are the main causes of climate change?",
            "Describe the process of photosynthesis.",
            "How does machine learning work?"
        ]
        
    def benchmark_single_generation(self, model: OllamaModel, prompt: str, 
                                  config: GenerationConfig) -> Dict[str, float]:
        """Benchmark a single generation."""
        start_time = time.time()
        
        # Generate response
        response = self.wrapper.generate(prompt, model=model, config=config, stream=False)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Count tokens (rough estimation)
        response_tokens = len(response.split())
        prompt_tokens = len(prompt.split())
        total_tokens = prompt_tokens + response_tokens
        
        return {
            'generation_time': generation_time,
            'prompt_tokens': prompt_tokens,
            'response_tokens': response_tokens,
            'total_tokens': total_tokens,
            'tokens_per_second': response_tokens / generation_time if generation_time > 0 else 0,
            'response_length': len(response)
        }
        
    def benchmark_streaming(self, model: OllamaModel, prompt: str,
                          config: GenerationConfig) -> Dict[str, float]:
        """Benchmark streaming generation."""
        tokens = []
        start_time = time.time()
        first_token_time = None
        
        # Stream response
        for i, token in enumerate(self.wrapper.generate(prompt, model=model, 
                                                       config=config, stream=True)):
            if i == 0 and first_token_time is None:
                first_token_time = time.time()
            tokens.append(token)
            
        end_time = time.time()
        
        total_time = end_time - start_time
        time_to_first_token = first_token_time - start_time if first_token_time else 0
        
        response = ''.join(tokens)
        token_count = len(response.split())
        
        return {
            'total_time': total_time,
            'time_to_first_token': time_to_first_token,
            'tokens_generated': token_count,
            'tokens_per_second': token_count / total_time if total_time > 0 else 0,
            'streaming_overhead': time_to_first_token / total_time if total_time > 0 else 0
        }
        
    def benchmark_model(self, model: OllamaModel, num_runs: int = 3) -> Dict[str, any]:
        """Comprehensive benchmark of a model."""
        print(f"\nBenchmarking {model.value}...")
        
        # Test different configurations
        configs = [
            ("small", GenerationConfig(max_tokens=50, temperature=0.1)),
            ("medium", GenerationConfig(max_tokens=200, temperature=0.7)),
            ("large", GenerationConfig(max_tokens=500, temperature=0.9))
        ]
        
        results = {
            'model': model.value,
            'configurations': {}
        }
        
        for config_name, config in configs:
            print(f"  Testing {config_name} configuration...")
            
            config_results = {
                'generation_times': [],
                'tokens_per_second': [],
                'response_tokens': []
            }
            
            # Run multiple times for each prompt
            for prompt in self.test_prompts[:num_runs]:
                bench_result = self.benchmark_single_generation(model, prompt, config)
                config_results['generation_times'].append(bench_result['generation_time'])
                config_results['tokens_per_second'].append(bench_result['tokens_per_second'])
                config_results['response_tokens'].append(bench_result['response_tokens'])
                
            # Calculate statistics
            results['configurations'][config_name] = {
                'avg_generation_time': statistics.mean(config_results['generation_times']),
                'avg_tokens_per_second': statistics.mean(config_results['tokens_per_second']),
                'avg_response_tokens': statistics.mean(config_results['response_tokens']),
                'std_tokens_per_second': statistics.stdev(config_results['tokens_per_second']) 
                    if len(config_results['tokens_per_second']) > 1 else 0
            }
            
        # Test streaming for medium config
        print("  Testing streaming performance...")
        streaming_results = []
        for prompt in self.test_prompts[:2]:
            stream_result = self.benchmark_streaming(model, prompt, configs[1][1])
            streaming_results.append(stream_result)
            
        results['streaming'] = {
            'avg_time_to_first_token': statistics.mean([r['time_to_first_token'] 
                                                       for r in streaming_results]),
            'avg_tokens_per_second': statistics.mean([r['tokens_per_second'] 
                                                    for r in streaming_results])
        }
        
        return results
        
    def run_full_benchmark(self) -> Dict[str, any]:
        """Run benchmark on all models."""
        all_results = {}
        
        for model in OllamaModel:
            try:
                model_results = self.benchmark_model(model)
                all_results[model.value] = model_results
                
                # Print summary
                print(f"\n{model.value} Summary:")
                for config_name, stats in model_results['configurations'].items():
                    print(f"  {config_name}: {stats['avg_tokens_per_second']:.1f} tokens/s")
                    
            except Exception as e:
                print(f"Error benchmarking {model.value}: {e}")
                all_results[model.value] = {'error': str(e)}
                
        return all_results
        
    def compare_models(self, results: Dict[str, any]):
        """Compare model performance."""
        print("\n=== Model Comparison ===")
        print("\nAverage Tokens/Second (medium config):")
        
        model_speeds = []
        for model_name, model_results in results.items():
            if 'error' not in model_results:
                medium_speed = model_results['configurations']['medium']['avg_tokens_per_second']
                model_speeds.append((model_name, medium_speed))
                
        # Sort by speed
        model_speeds.sort(key=lambda x: x[1], reverse=True)
        
        for model, speed in model_speeds:
            print(f"  {model}: {speed:.1f} tokens/s")
            
        print("\nStreaming Performance (time to first token):")
        for model_name, model_results in results.items():
            if 'error' not in model_results and 'streaming' in model_results:
                ttft = model_results['streaming']['avg_time_to_first_token']
                print(f"  {model}: {ttft*1000:.0f}ms")

def main():
    """Run Ollama benchmarks."""
    print("=== Ollama Token Generation Benchmark ===")
    
    benchmark = OllamaBenchmark()
    
    # Quick test to ensure Ollama is running
    try:
        models = benchmark.wrapper.list_models()
        print(f"Found {len(models)} models")
    except Exception as e:
        print(f"Error: Cannot connect to Ollama. Make sure it's running.")
        print(f"Error details: {e}")
        return
        
    # Run benchmarks
    results = benchmark.run_full_benchmark()
    
    # Compare models
    benchmark.compare_models(results)
    
    # Save results
    with open('training/bootstrap/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n✅ Benchmark complete! Results saved to benchmark_results.json")
    
    # Print target achievement
    print("\n=== Performance vs Targets ===")
    targets = {
        "llama3.1:8b": (40, 50),  # min, max tokens/s
        "gemma3:27b": (25, 35),
        "qwen3:32b": (12, 20)
    }
    
    for model, (min_target, max_target) in targets.items():
        if model in results and 'error' not in results[model]:
            actual = results[model]['configurations']['medium']['avg_tokens_per_second']
            if min_target <= actual <= max_target:
                status = "✅ ACHIEVED"
            elif actual > max_target:
                status = "🚀 EXCEEDED"
            else:
                status = "❌ BELOW TARGET"
            print(f"{model}: {actual:.1f} tokens/s {status} (target: {min_target}-{max_target})")

if __name__ == "__main__":
    main()