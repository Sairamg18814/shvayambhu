"""Synthetic data generation pipeline for self-training.

This module provides the main synthetic data generation functionality
including text generation, domain-specific generation, and data augmentation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Iterator, Union
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
from collections import defaultdict
import hashlib

from ...core.blt.pipeline import BLTPipeline
from ...core.blt.entropy import calculate_byte_entropy
from .quality_filter import QualityFilter, QualityMetrics
from .diversity import DiversityChecker, DiversityMetrics


@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation."""
    # Generation parameters
    max_length: int = 2048
    min_length: int = 64
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    
    # Domain settings
    domains: List[str] = field(default_factory=lambda: [
        "general", "code", "technical", "creative", "academic"
    ])
    domain_weights: Dict[str, float] = field(default_factory=lambda: {
        "general": 0.3,
        "code": 0.2,
        "technical": 0.2,
        "creative": 0.15,
        "academic": 0.15
    })
    
    # Quality thresholds
    min_quality_score: float = 0.7
    min_diversity_score: float = 0.6
    max_repetition_ratio: float = 0.2
    
    # Batch settings
    batch_size: int = 32
    num_generations: int = 10000
    
    # Output settings
    output_dir: str = "synthetic_data/"
    save_frequency: int = 1000
    format: str = "jsonl"  # jsonl, parquet, or txt


@dataclass
class GenerationStats:
    """Statistics for generation process."""
    total_generated: int = 0
    accepted: int = 0
    rejected_quality: int = 0
    rejected_diversity: int = 0
    rejected_repetition: int = 0
    domain_counts: Dict[str, int] = field(default_factory=dict)
    avg_quality_score: float = 0.0
    avg_diversity_score: float = 0.0
    avg_length: float = 0.0
    generation_time: float = 0.0


class SyntheticDataGenerator:
    """Main synthetic data generation pipeline."""
    
    def __init__(
        self,
        model: BLTPipeline,
        config: GenerationConfig,
        quality_filter: Optional[QualityFilter] = None,
        diversity_checker: Optional[DiversityChecker] = None
    ):
        self.model = model
        self.config = config
        self.quality_filter = quality_filter or QualityFilter()
        self.diversity_checker = diversity_checker or DiversityChecker()
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generation history
        self.generated_hashes = set()
        self.stats = GenerationStats()
        
        # Domain-specific generators
        self.domain_generators = self._setup_domain_generators()
    
    def _setup_domain_generators(self) -> Dict[str, 'DomainSpecificGenerator']:
        """Setup domain-specific generators."""
        generators = {}
        for domain in self.config.domains:
            generators[domain] = DomainSpecificGenerator(
                domain=domain,
                model=self.model,
                config=self.config
            )
        return generators
    
    def generate_dataset(
        self,
        num_samples: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate synthetic dataset."""
        num_samples = num_samples or self.config.num_generations
        save_path = save_path or self.output_dir / f"synthetic_{int(time.time())}.{self.config.format}"
        
        dataset = []
        batch_buffer = []
        start_time = time.time()
        
        # Set model to eval mode
        self.model.eval()
        
        with tqdm(total=num_samples, desc="Generating synthetic data") as pbar:
            while self.stats.accepted < num_samples:
                # Select domain
                domain = self._select_domain()
                
                # Generate batch
                batch = self._generate_batch(domain)
                
                # Process and filter batch
                for sample in batch:
                    if self._validate_sample(sample):
                        dataset.append(sample)
                        batch_buffer.append(sample)
                        self.stats.accepted += 1
                        pbar.update(1)
                        
                        # Save periodically
                        if len(batch_buffer) >= self.config.save_frequency:
                            self._save_batch(batch_buffer, save_path)
                            batch_buffer = []
                
                # Update progress
                pbar.set_postfix({
                    "accepted": self.stats.accepted,
                    "quality_reject": self.stats.rejected_quality,
                    "diversity_reject": self.stats.rejected_diversity
                })
        
        # Save remaining samples
        if batch_buffer:
            self._save_batch(batch_buffer, save_path)
        
        # Update statistics
        self.stats.generation_time = time.time() - start_time
        self._save_statistics()
        
        return dataset
    
    def _select_domain(self) -> str:
        """Select domain based on weights."""
        domains = list(self.config.domain_weights.keys())
        weights = list(self.config.domain_weights.values())
        return np.random.choice(domains, p=weights)
    
    def _generate_batch(self, domain: str) -> List[Dict[str, Any]]:
        """Generate a batch of samples for a domain."""
        generator = self.domain_generators[domain]
        samples = []
        
        for _ in range(self.config.batch_size):
            # Generate sample
            text = generator.generate()
            
            # Create sample dict
            sample = {
                "text": text,
                "domain": domain,
                "metadata": {
                    "temperature": self.config.temperature,
                    "top_k": self.config.top_k,
                    "top_p": self.config.top_p,
                    "length": len(text),
                    "timestamp": time.time()
                }
            }
            
            samples.append(sample)
            self.stats.total_generated += 1
            
            # Update domain counts
            if domain not in self.stats.domain_counts:
                self.stats.domain_counts[domain] = 0
            self.stats.domain_counts[domain] += 1
        
        return samples
    
    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate a generated sample."""
        text = sample["text"]
        
        # Check length
        if len(text) < self.config.min_length:
            return False
        
        # Check for duplicates
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in self.generated_hashes:
            self.stats.rejected_diversity += 1
            return False
        
        # Quality check
        quality_metrics = self.quality_filter.evaluate(text)
        if quality_metrics.overall_score < self.config.min_quality_score:
            self.stats.rejected_quality += 1
            return False
        
        # Diversity check
        diversity_metrics = self.diversity_checker.check_diversity(
            text, self.generated_hashes
        )
        if diversity_metrics.diversity_score < self.config.min_diversity_score:
            self.stats.rejected_diversity += 1
            return False
        
        # Repetition check
        if quality_metrics.repetition_ratio > self.config.max_repetition_ratio:
            self.stats.rejected_repetition += 1
            return False
        
        # Update sample with metrics
        sample["quality_metrics"] = quality_metrics.__dict__
        sample["diversity_metrics"] = diversity_metrics.__dict__
        
        # Add to hash set
        self.generated_hashes.add(text_hash)
        
        # Update statistics
        self.stats.avg_quality_score = (
            (self.stats.avg_quality_score * (self.stats.accepted - 1) + 
             quality_metrics.overall_score) / max(self.stats.accepted, 1)
        )
        self.stats.avg_diversity_score = (
            (self.stats.avg_diversity_score * (self.stats.accepted - 1) + 
             diversity_metrics.diversity_score) / max(self.stats.accepted, 1)
        )
        self.stats.avg_length = (
            (self.stats.avg_length * (self.stats.accepted - 1) + 
             len(text)) / max(self.stats.accepted, 1)
        )
        
        return True
    
    def _save_batch(self, batch: List[Dict[str, Any]], save_path: Path):
        """Save a batch of samples."""
        if self.config.format == "jsonl":
            with open(save_path, 'a') as f:
                for sample in batch:
                    f.write(json.dumps(sample) + '\n')
        elif self.config.format == "parquet":
            import pandas as pd
            df = pd.DataFrame(batch)
            if save_path.exists():
                existing_df = pd.read_parquet(save_path)
                df = pd.concat([existing_df, df], ignore_index=True)
            df.to_parquet(save_path, index=False)
        else:  # txt format
            with open(save_path, 'a') as f:
                for sample in batch:
                    f.write(sample["text"] + '\n\n')
    
    def _save_statistics(self):
        """Save generation statistics."""
        stats_path = self.output_dir / "generation_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats.__dict__, f, indent=2)
    
    def augment_data(
        self,
        input_texts: List[str],
        augmentation_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Augment existing data with variations."""
        if augmentation_types is None:
            augmentation_types = ["paraphrase", "continuation", "infilling", "summarization"]
        
        augmented_samples = []
        
        for text in tqdm(input_texts, desc="Augmenting data"):
            for aug_type in augmentation_types:
                augmented = self._apply_augmentation(text, aug_type)
                if augmented and self._validate_sample({"text": augmented}):
                    sample = {
                        "text": augmented,
                        "original_text": text,
                        "augmentation_type": aug_type,
                        "domain": "augmented",
                        "metadata": {
                            "timestamp": time.time()
                        }
                    }
                    augmented_samples.append(sample)
        
        return augmented_samples
    
    def _apply_augmentation(self, text: str, aug_type: str) -> Optional[str]:
        """Apply specific augmentation to text."""
        if aug_type == "paraphrase":
            prompt = f"Paraphrase the following text:\n{text}\n\nParaphrase:"
        elif aug_type == "continuation":
            prompt = text[:len(text)//2]  # Use first half as prompt
        elif aug_type == "infilling":
            # Create masked version
            words = text.split()
            mask_indices = np.random.choice(
                len(words), size=max(1, len(words)//5), replace=False
            )
            for idx in mask_indices:
                words[idx] = "[MASK]"
            prompt = " ".join(words)
        elif aug_type == "summarization":
            prompt = f"Summarize the following text:\n{text}\n\nSummary:"
        else:
            return None
        
        # Generate augmented text
        with torch.no_grad():
            # Convert prompt to bytes
            prompt_bytes = torch.tensor(
                list(prompt.encode('utf-8')), dtype=torch.uint8
            ).unsqueeze(0)
            
            # Generate
            generated_bytes = self._generate_from_prompt(prompt_bytes)
            
            # Convert back to text
            try:
                augmented_text = bytes(generated_bytes.cpu().numpy()).decode('utf-8')
                return augmented_text
            except:
                return None
    
    @torch.no_grad()
    def _generate_from_prompt(
        self,
        prompt: torch.Tensor,
        max_new_tokens: Optional[int] = None
    ) -> torch.Tensor:
        """Generate text from prompt."""
        max_new_tokens = max_new_tokens or self.config.max_length - prompt.shape[1]
        device = next(self.model.parameters()).device
        prompt = prompt.to(device)
        
        generated = prompt.clone()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Get model predictions
            outputs = self.model(
                generated if past_key_values is None else generated[:, -1:],
                mode='inference',
                past_key_values=past_key_values
            )
            
            logits = outputs.get('logits', outputs.get('patch_logits'))
            if logits is None:
                break
            
            # Get next token logits
            next_logits = logits[:, -1, :] / self.config.temperature
            
            # Apply repetition penalty
            if self.config.repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    next_logits[0, token_id] /= self.config.repetition_penalty
            
            # Apply top-k filtering
            if self.config.top_k > 0:
                top_k_values, top_k_indices = torch.topk(
                    next_logits, min(self.config.top_k, next_logits.shape[-1])
                )
                next_logits = torch.full_like(next_logits, -float('inf'))
                next_logits.scatter_(1, top_k_indices, top_k_values)
            
            # Apply top-p filtering
            if self.config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                
                sorted_indices_to_remove = cumulative_probs > self.config.top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = -float('inf')
            
            # Sample next token
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update past key values if available
            if 'past_key_values' in outputs:
                past_key_values = outputs['past_key_values']
            
            # Check for end token (0)
            if next_token.item() == 0:
                break
        
        return generated[0]


class DomainSpecificGenerator:
    """Domain-specific text generation."""
    
    def __init__(
        self,
        domain: str,
        model: BLTPipeline,
        config: GenerationConfig
    ):
        self.domain = domain
        self.model = model
        self.config = config
        
        # Domain-specific prompts and styles
        self.domain_prompts = self._get_domain_prompts()
        self.domain_styles = self._get_domain_styles()
    
    def _get_domain_prompts(self) -> List[str]:
        """Get domain-specific prompts."""
        prompts = {
            "general": [
                "The importance of",
                "In today's world,",
                "One of the most",
                "Research has shown that",
                "It is widely known that"
            ],
            "code": [
                "def ",
                "class ",
                "import ",
                "# This function",
                "// Implementation of"
            ],
            "technical": [
                "The system architecture",
                "Technical specifications:",
                "Algorithm description:",
                "Performance analysis shows",
                "Implementation details:"
            ],
            "creative": [
                "Once upon a time,",
                "The story begins with",
                "In a world where",
                "She looked at",
                "The sound of"
            ],
            "academic": [
                "Abstract: This paper",
                "Introduction: The study of",
                "Methods: We conducted",
                "Results indicate that",
                "In conclusion,"
            ]
        }
        return prompts.get(self.domain, prompts["general"])
    
    def _get_domain_styles(self) -> Dict[str, Any]:
        """Get domain-specific generation parameters."""
        styles = {
            "general": {
                "temperature": 0.8,
                "top_p": 0.9,
                "repetition_penalty": 1.2
            },
            "code": {
                "temperature": 0.6,
                "top_p": 0.95,
                "repetition_penalty": 1.0
            },
            "technical": {
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            },
            "creative": {
                "temperature": 0.9,
                "top_p": 0.95,
                "repetition_penalty": 1.3
            },
            "academic": {
                "temperature": 0.7,
                "top_p": 0.85,
                "repetition_penalty": 1.2
            }
        }
        return styles.get(self.domain, styles["general"])
    
    def generate(self) -> str:
        """Generate domain-specific text."""
        # Select random prompt
        prompt = np.random.choice(self.domain_prompts)
        
        # Get domain-specific parameters
        style = self.domain_styles
        
        # Update config temporarily
        original_temp = self.config.temperature
        original_top_p = self.config.top_p
        original_rep_penalty = self.config.repetition_penalty
        
        self.config.temperature = style["temperature"]
        self.config.top_p = style["top_p"]
        self.config.repetition_penalty = style["repetition_penalty"]
        
        # Generate text
        prompt_bytes = torch.tensor(
            list(prompt.encode('utf-8')), dtype=torch.uint8
        ).unsqueeze(0)
        
        generator = SyntheticDataGenerator(self.model, self.config)
        generated_bytes = generator._generate_from_prompt(prompt_bytes)
        
        # Restore original config
        self.config.temperature = original_temp
        self.config.top_p = original_top_p
        self.config.repetition_penalty = original_rep_penalty
        
        # Convert to text
        try:
            text = bytes(generated_bytes.cpu().numpy()).decode('utf-8')
            return text
        except:
            return prompt  # Fallback to prompt if decoding fails
    
    def generate_structured(self, structure_type: str) -> str:
        """Generate structured domain-specific content."""
        if self.domain == "code":
            return self._generate_code_structure(structure_type)
        elif self.domain == "technical":
            return self._generate_technical_structure(structure_type)
        elif self.domain == "academic":
            return self._generate_academic_structure(structure_type)
        else:
            return self.generate()  # Default generation
    
    def _generate_code_structure(self, structure_type: str) -> str:
        """Generate structured code."""
        structures = {
            "function": "def {name}({params}):\n    '''{docstring}'''\n    {body}",
            "class": "class {name}:\n    '''{docstring}'''\n    \n    def __init__(self, {params}):\n        {init_body}",
            "module": "#!/usr/bin/env python3\n'''{docstring}'''\n\nimport {imports}\n\n{content}"
        }
        
        template = structures.get(structure_type, structures["function"])
        
        # Generate components
        components = {}
        for placeholder in ["{name}", "{params}", "{docstring}", "{body}", 
                           "{init_body}", "{imports}", "{content}"]:
            if placeholder in template:
                # Generate content for placeholder
                prompt = f"Generate {placeholder.strip('{}')}: "
                generated = self.generate()
                components[placeholder] = generated.split('\n')[0]  # First line
        
        # Fill template
        result = template
        for placeholder, value in components.items():
            result = result.replace(placeholder, value)
        
        return result
    
    def _generate_technical_structure(self, structure_type: str) -> str:
        """Generate structured technical content."""
        structures = {
            "specification": "# {title}\n\n## Overview\n{overview}\n\n## Requirements\n{requirements}\n\n## Implementation\n{implementation}",
            "documentation": "# {title}\n\n## Description\n{description}\n\n## Usage\n{usage}\n\n## Examples\n{examples}",
            "report": "# {title}\n\n## Executive Summary\n{summary}\n\n## Analysis\n{analysis}\n\n## Recommendations\n{recommendations}"
        }
        
        template = structures.get(structure_type, structures["documentation"])
        
        # Generate sections
        sections = {}
        for match in ["{title}", "{overview}", "{requirements}", "{implementation}",
                     "{description}", "{usage}", "{examples}", "{summary}",
                     "{analysis}", "{recommendations}"]:
            if match in template:
                prompt = f"Technical {match.strip('{}')}: "
                generated = self.generate()
                sections[match] = generated
        
        # Fill template
        result = template
        for placeholder, value in sections.items():
            result = result.replace(placeholder, value)
        
        return result
    
    def _generate_academic_structure(self, structure_type: str) -> str:
        """Generate structured academic content."""
        structures = {
            "abstract": "Abstract: {introduction} {methods} {results} {conclusion}",
            "introduction": "Introduction: {background} {problem} {contribution} {outline}",
            "methodology": "Methodology: {approach} {data} {experiments} {evaluation}"
        }
        
        template = structures.get(structure_type, structures["abstract"])
        
        # Generate components
        components = {}
        for placeholder in template.split():
            if placeholder.startswith("{") and placeholder.endswith("}"):
                component = placeholder.strip("{}")
                prompt = f"Academic paper {component}: "
                generated = self.generate()
                components[placeholder] = generated.split('.')[0] + "."
        
        # Fill template
        result = template
        for placeholder, value in components.items():
            result = result.replace(placeholder, value)
        
        return result