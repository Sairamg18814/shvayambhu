"""Simple Ollama benchmark script."""

import time
import json
import requests
import statistics
from typing import Dict, List

class OllamaBenchmark:
    """Simple Ollama benchmark without complex imports."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.models = ["llama3.1:8b", "gemma3:27b", "Qwen3:32b"]
        self.test_prompts = [
            "Explain quantum computing in simple terms.",
            "Write a Python function to sort a list.",
            "What are the main causes of climate change?"
        ]
        
    def generate(self, model: str, prompt: str, max_tokens: int = 200) -> Dict:
        """Generate text using Ollama API."""
        payload = {
            "model": model,
            "prompt": prompt,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            },
            "stream": False
        }
        
        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=300
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            generation_time = end_time - start_time
            response_text = data.get("response", "")
            
            # Rough token count
            tokens = len(response_text.split())
            
            return {
                "success": True,
                "generation_time": generation_time,
                "response": response_text,
                "tokens": tokens,
                "tokens_per_second": tokens / generation_time if generation_time > 0 else 0
            }
        else:
            return {"success": False, "error": response.text}
            
    def benchmark_model(self, model: str) -> Dict:
        """Benchmark a single model."""
        print(f"\nBenchmarking {model}...")
        
        results = {
            "model": model,
            "runs": [],
            "errors": 0
        }
        
        for i, prompt in enumerate(self.test_prompts):
            print(f"  Run {i+1}/{len(self.test_prompts)}...", end="", flush=True)
            
            result = self.generate(model, prompt)
            
            if result["success"]:
                results["runs"].append({
                    "prompt": prompt,
                    "generation_time": result["generation_time"],
                    "tokens": result["tokens"],
                    "tokens_per_second": result["tokens_per_second"]
                })
                print(f" {result['tokens_per_second']:.1f} tokens/s")
            else:
                results["errors"] += 1
                print(" ERROR")
                
        # Calculate statistics
        if results["runs"]:
            speeds = [r["tokens_per_second"] for r in results["runs"]]
            results["statistics"] = {
                "avg_tokens_per_second": statistics.mean(speeds),
                "min_tokens_per_second": min(speeds),
                "max_tokens_per_second": max(speeds),
                "std_tokens_per_second": statistics.stdev(speeds) if len(speeds) > 1 else 0
            }
            
        return results
        
    def run_benchmark(self) -> Dict:
        """Run benchmark on all models."""
        print("=== Ollama Token Generation Speed Benchmark ===")
        
        # Check connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                print("Error: Cannot connect to Ollama")
                return {}
        except:
            print("Error: Ollama is not running. Start it with 'ollama serve'")
            return {}
            
        all_results = {}
        
        for model in self.models:
            results = self.benchmark_model(model)
            all_results[model] = results
            
        return all_results
        
    def print_summary(self, results: Dict):
        """Print benchmark summary."""
        print("\n=== Benchmark Summary ===")
        print("\nAverage Token Generation Speed:")
        
        for model, data in results.items():
            if "statistics" in data:
                avg_speed = data["statistics"]["avg_tokens_per_second"]
                print(f"  {model}: {avg_speed:.1f} tokens/s")
            else:
                print(f"  {model}: ERROR")
                
        print("\n=== Performance vs Targets ===")
        targets = {
            "llama3.1:8b": (40, 50),
            "gemma3:27b": (25, 35),
            "Qwen3:32b": (12, 20)
        }
        
        for model, (min_target, max_target) in targets.items():
            if model in results and "statistics" in results[model]:
                actual = results[model]["statistics"]["avg_tokens_per_second"]
                if min_target <= actual <= max_target:
                    status = "✅ ACHIEVED"
                elif actual > max_target:
                    status = "🚀 EXCEEDED"
                else:
                    status = "❌ BELOW TARGET"
                print(f"{model}: {actual:.1f} tokens/s {status} (target: {min_target}-{max_target})")

def main():
    benchmark = OllamaBenchmark()
    results = benchmark.run_benchmark()
    
    if results:
        benchmark.print_summary(results)
        
        # Save results
        with open("scripts/ollama_benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        print("\n✅ Results saved to ollama_benchmark_results.json")

if __name__ == "__main__":
    main()