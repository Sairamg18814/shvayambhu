"""Ollama API wrapper for model integration."""

import os
import json
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Generator, Union
from dataclasses import dataclass
from enum import Enum
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.error_handling import (
    with_error_handling, retry_with_backoff, ModelError
)
from utils.logging_config import get_logger

logger = get_logger(__name__)

class OllamaModel(Enum):
    """Available Ollama models."""
    QWEN3_32B = "qwen3:32b"
    GEMMA3_27B = "gemma3:27b"
    LLAMA3_1_8B = "llama3.1:8b"

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    repeat_penalty: float = 1.1
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    
@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    size: int
    digest: str
    modified_at: str
    parameters: Dict[str, Any]

class OllamaWrapper:
    """Wrapper for Ollama API with retry logic and error handling."""
    
    def __init__(self, base_url: Optional[str] = None, timeout: int = 300):
        self.base_url = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout
        
        # Configure session with retry
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Model switching state
        self.current_model: Optional[OllamaModel] = None
        self._model_cache: Dict[str, ModelInfo] = {}
        
        logger.info(f"Ollama wrapper initialized with base URL: {self.base_url}")
        
    @with_error_handling(exceptions=(requests.RequestException, ModelError))
    def list_models(self) -> List[ModelInfo]:
        """List all available models."""
        response = self.session.get(
            f"{self.base_url}/api/tags",
            timeout=self.timeout
        )
        response.raise_for_status()
        
        models = []
        for model_data in response.json().get("models", []):
            models.append(ModelInfo(
                name=model_data["name"],
                size=model_data["size"],
                digest=model_data["digest"],
                modified_at=model_data["modified_at"],
                parameters=model_data.get("details", {})
            ))
            
        return models
    
    @with_error_handling(exceptions=(requests.RequestException, ModelError))
    def model_info(self, model: Union[OllamaModel, str]) -> ModelInfo:
        """Get information about a specific model."""
        model_name = model.value if isinstance(model, OllamaModel) else model
        
        # Check cache first
        if model_name in self._model_cache:
            return self._model_cache[model_name]
            
        response = self.session.post(
            f"{self.base_url}/api/show",
            json={"name": model_name},
            timeout=self.timeout
        )
        response.raise_for_status()
        
        data = response.json()
        info = ModelInfo(
            name=data["name"],
            size=data.get("size", 0),
            digest=data.get("digest", ""),
            modified_at=data.get("modified_at", ""),
            parameters=data.get("parameters", {})
        )
        
        self._model_cache[model_name] = info
        return info
    
    @retry_with_backoff(max_attempts=3, exceptions=(requests.RequestException,))
    def generate(
        self,
        prompt: str,
        model: Optional[Union[OllamaModel, str]] = None,
        config: Optional[GenerationConfig] = None,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate text using specified model.
        
        Args:
            prompt: Input prompt
            model: Model to use (defaults to current model)
            config: Generation configuration
            stream: Whether to stream the response
            
        Returns:
            Generated text or generator for streaming
        """
        if model is None:
            model = self.current_model or OllamaModel.LLAMA3_1_8B
            
        model_name = model.value if isinstance(model, OllamaModel) else model
        config = config or GenerationConfig()
        
        # Update current model
        if isinstance(model, OllamaModel):
            self.current_model = model
            
        payload = {
            "model": model_name,
            "prompt": prompt,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "num_predict": config.max_tokens,
                "repeat_penalty": config.repeat_penalty,
            },
            "stream": stream
        }
        
        if config.seed is not None:
            payload["options"]["seed"] = config.seed
            
        if config.stop:
            payload["options"]["stop"] = config.stop
            
        logger.info(f"Generating with model {model_name}, stream={stream}")
        
        response = self.session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=stream,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        if stream:
            return self._stream_response(response)
        else:
            return response.json()["response"]
    
    def _stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        """Stream response tokens."""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]
                    
                if data.get("done", False):
                    break
    
    @retry_with_backoff(max_attempts=3)
    async def generate_async(
        self,
        prompt: str,
        model: Optional[Union[OllamaModel, str]] = None,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Async version of generate."""
        import aiohttp
        
        model_name = (model.value if isinstance(model, OllamaModel) else model) or \
                     (self.current_model.value if self.current_model else OllamaModel.LLAMA3_1_8B.value)
        config = config or GenerationConfig()
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "num_predict": config.max_tokens,
                "repeat_penalty": config.repeat_penalty,
            },
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data["response"]
    
    def switch_model(self, model: OllamaModel) -> None:
        """Switch the current active model."""
        logger.info(f"Switching from {self.current_model} to {model}")
        self.current_model = model
        
    @with_error_handling(exceptions=(requests.RequestException,))
    def test_model(self, model: Union[OllamaModel, str]) -> Dict[str, Any]:
        """Test a model with a simple prompt."""
        test_prompt = "Hello, please respond with 'Model is working' if you can see this."
        
        start_time = time.time()
        response = self.generate(
            prompt=test_prompt,
            model=model,
            config=GenerationConfig(max_tokens=50, temperature=0.1)
        )
        end_time = time.time()
        
        return {
            "model": model.value if isinstance(model, OllamaModel) else model,
            "response": response,
            "response_time": end_time - start_time,
            "working": "working" in response.lower()
        }
    
    def benchmark_models(self, prompt: str = None) -> Dict[str, Dict[str, Any]]:
        """Benchmark all available models."""
        prompt = prompt or "Explain quantum computing in one paragraph."
        results = {}
        
        for model in OllamaModel:
            logger.info(f"Benchmarking {model.value}")
            
            try:
                config = GenerationConfig(max_tokens=200, temperature=0.7)
                
                # Measure generation time
                start_time = time.time()
                response = self.generate(prompt, model=model, config=config)
                end_time = time.time()
                
                # Calculate tokens per second
                response_tokens = len(response.split())
                tokens_per_second = response_tokens / (end_time - start_time)
                
                results[model.value] = {
                    "response_length": len(response),
                    "response_tokens": response_tokens,
                    "generation_time": end_time - start_time,
                    "tokens_per_second": tokens_per_second,
                    "success": True
                }
                
            except Exception as e:
                logger.error(f"Error benchmarking {model.value}: {e}")
                results[model.value] = {
                    "error": str(e),
                    "success": False
                }
                
        return results
    
    def create_embeddings(
        self,
        text: str,
        model: Optional[Union[OllamaModel, str]] = None
    ) -> List[float]:
        """Create embeddings for text using specified model."""
        model_name = model.value if isinstance(model, OllamaModel) else model
        model_name = model_name or (self.current_model.value if self.current_model else OllamaModel.LLAMA3_1_8B.value)
        
        response = self.session.post(
            f"{self.base_url}/api/embeddings",
            json={"model": model_name, "prompt": text},
            timeout=self.timeout
        )
        response.raise_for_status()
        
        return response.json()["embedding"]
    
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.session.close()


# Convenience functions
def create_ollama_client() -> OllamaWrapper:
    """Create a configured Ollama client."""
    return OllamaWrapper()

def test_ollama_connection() -> bool:
    """Test if Ollama is accessible."""
    try:
        client = OllamaWrapper()
        models = client.list_models()
        logger.info(f"Ollama connection successful. Found {len(models)} models.")
        return True
    except Exception as e:
        logger.error(f"Ollama connection failed: {e}")
        return False