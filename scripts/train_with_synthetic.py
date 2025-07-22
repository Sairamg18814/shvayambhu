#!/usr/bin/env python3
"""Training script with synthetic data generation.

This script demonstrates the complete training pipeline including
synthetic data generation, quality filtering, and model training.
"""

import torch
import argparse
from pathlib import Path
import json
import time
from typing import Dict, Any

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.blt.pipeline import BLTPipeline
from training.synthetic import (
    SyntheticDataGenerator,
    GenerationConfig,
    QualityFilter,
    DiversityChecker
)
from training.bootstrap import (
    BootstrapTrainingLoop,
    TrainingConfig,
    BootstrapDataLoader,
    MultiObjectiveTrainer,
    ValidationPipeline,
    ValidationConfig
)
from training.bootstrap.statistics_tracker import StatisticsTracker


def setup_model(config: Dict[str, Any]) -> BLTPipeline:
    """Setup BLT model for training."""
    model_config = {
        'vocab_size': 256,  # Byte vocabulary
        'hidden_dim': config.get('hidden_dim', 768),
        'num_layers': config.get('num_layers', 12),
        'num_heads': config.get('num_heads', 12),
        'intermediate_size': config.get('intermediate_size', 3072),
        'max_position_embeddings': config.get('max_position_embeddings', 2048),
        'patch_size': config.get('patch_size', 16),
        'use_rope': True,
        'use_swiglu': True
    }
    
    model = BLTPipeline(model_config)
    
    # Initialize weights
    model.apply(lambda m: torch.nn.init.normal_(m.weight, std=0.02) 
                if hasattr(m, 'weight') and m.weight.dim() > 1 else None)
    
    return model


def generate_synthetic_data(
    model: BLTPipeline,
    config: Dict[str, Any]
) -> Path:
    """Generate synthetic training data."""
    print("Generating synthetic data...")
    
    # Setup generation config
    gen_config = GenerationConfig(
        max_length=config.get('max_gen_length', 1024),
        min_length=config.get('min_gen_length', 64),
        temperature=config.get('temperature', 0.8),
        top_k=config.get('top_k', 50),
        top_p=config.get('top_p', 0.9),
        batch_size=config.get('gen_batch_size', 16),
        num_generations=config.get('num_synthetic_samples', 5000),
        output_dir=config.get('synthetic_output_dir', 'synthetic_data'),
        domains=config.get('domains', ['general', 'code', 'technical'])
    )
    
    # Setup quality filter
    quality_filter = QualityFilter(
        min_length=gen_config.min_length,
        max_length=gen_config.max_length,
        min_vocabulary_diversity=0.3,
        max_repetition_ratio=0.2
    )
    
    # Setup diversity checker
    diversity_checker = DiversityChecker(
        min_similarity_threshold=0.2,
        max_similarity_threshold=0.8,
        n_reference_samples=100
    )
    
    # Create generator
    generator = SyntheticDataGenerator(
        model=model,
        config=gen_config,
        quality_filter=quality_filter,
        diversity_checker=diversity_checker
    )
    
    # Generate dataset
    start_time = time.time()
    dataset = generator.generate_dataset()
    generation_time = time.time() - start_time
    
    print(f"Generated {len(dataset)} samples in {generation_time:.2f}s")
    print(f"Acceptance rate: {generator.stats.accepted / generator.stats.total_generated:.3f}")
    
    # Save dataset path
    output_path = Path(gen_config.output_dir) / f"synthetic_{int(time.time())}.jsonl"
    
    return output_path


def train_model(
    model: BLTPipeline,
    synthetic_data_path: Path,
    config: Dict[str, Any]
) -> BLTPipeline:
    """Train model with synthetic data."""
    print("Training model with synthetic data...")
    
    # Setup training config
    training_config = TrainingConfig(
        learning_rate=config.get('learning_rate', 1e-4),
        batch_size=config.get('batch_size', 32),
        num_epochs=config.get('num_epochs', 3),
        warmup_steps=config.get('warmup_steps', 1000),
        gradient_accumulation_steps=config.get('gradient_accumulation', 4),
        max_grad_norm=config.get('max_grad_norm', 1.0),
        save_steps=config.get('save_steps', 1000),
        eval_steps=config.get('eval_steps', 500),
        output_dir=config.get('output_dir', 'checkpoints'),
        logging_steps=config.get('logging_steps', 100)
    )
    
    # Setup data loader
    data_loader = BootstrapDataLoader(
        batch_size=training_config.batch_size,
        max_length=config.get('max_length', 1024),
        num_workers=config.get('num_workers', 4)
    )
    
    # Load synthetic data
    data_loader.load_from_file(str(synthetic_data_path), format='jsonl')
    
    # Setup statistics tracker
    stats_tracker = StatisticsTracker(
        db_path=Path(training_config.output_dir) / 'training_stats.db'
    )
    
    # Setup validation
    validation_config = ValidationConfig(
        eval_batch_size=16,
        max_eval_samples=500,
        generation_max_length=512
    )
    
    validation_pipeline = ValidationPipeline(
        model=model,
        config=validation_config,
        data_loader=data_loader,
        statistics_tracker=stats_tracker
    )
    
    # Setup multi-objective trainer
    objective_trainer = MultiObjectiveTrainer(
        objectives=['next_byte', 'masked_modeling', 'entropy_prediction'],
        weights={'next_byte': 0.5, 'masked_modeling': 0.3, 'entropy_prediction': 0.2}
    )
    
    # Setup training loop
    training_loop = BootstrapTrainingLoop(
        model=model,
        config=training_config,
        data_loader=data_loader,
        objective_trainer=objective_trainer,
        validation_pipeline=validation_pipeline,
        statistics_tracker=stats_tracker
    )
    
    # Train the model
    trained_model = training_loop.train()
    
    print("Training completed!")
    
    return trained_model


def evaluate_model(model: BLTPipeline, config: Dict[str, Any]):
    """Evaluate the trained model."""
    print("Evaluating model...")
    
    # Setup validation config
    validation_config = ValidationConfig(
        eval_batch_size=16,
        max_eval_samples=1000,
        generation_max_length=512,
        eval_tasks=['next_byte_prediction', 'generation_quality']
    )
    
    # Create dummy validation data for testing
    validation_pipeline = ValidationPipeline(
        model=model,
        config=validation_config
    )
    
    # Run benchmarks
    benchmark_results = validation_pipeline.run_benchmark('comprehensive')
    
    print("Benchmark Results:")
    for metric, value in benchmark_results.items():
        print(f"  {metric}: {value:.4f}")
    
    return benchmark_results


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train model with synthetic data')
    parser.add_argument('--config', type=str, default='configs/training_config.json',
                       help='Path to training configuration file')
    parser.add_argument('--model-size', type=str, default='small',
                       choices=['small', 'medium', 'large'],
                       help='Model size preset')
    parser.add_argument('--synthetic-only', action='store_true',
                       help='Only generate synthetic data, skip training')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate existing model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to existing checkpoint to continue from')
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'hidden_dim': 768,
            'num_layers': 12,
            'num_heads': 12,
            'max_gen_length': 1024,
            'num_synthetic_samples': 10000,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 3,
            'output_dir': 'outputs'
        }
    
    # Model size presets
    size_configs = {
        'small': {'hidden_dim': 512, 'num_layers': 8, 'num_heads': 8},
        'medium': {'hidden_dim': 768, 'num_layers': 12, 'num_heads': 12},
        'large': {'hidden_dim': 1024, 'num_layers': 16, 'num_heads': 16}
    }
    
    if args.model_size in size_configs:
        config.update(size_configs[args.model_size])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup model
    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model = torch.load(args.checkpoint, map_location=device)
    else:
        model = setup_model(config)
    
    model = model.to(device)
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    if args.eval_only:
        # Only run evaluation
        benchmark_results = evaluate_model(model, config)
        
        # Save results
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        return
    
    # Generate synthetic data
    synthetic_data_path = generate_synthetic_data(model, config)
    
    if args.synthetic_only:
        print(f"Synthetic data generated at: {synthetic_data_path}")
        return
    
    # Train model
    trained_model = train_model(model, synthetic_data_path, config)
    
    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    torch.save(trained_model.state_dict(), final_model_path)
    print(f"Model saved to: {final_model_path}")
    
    # Evaluate trained model
    benchmark_results = evaluate_model(trained_model, config)
    
    # Save evaluation results
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    print("Training and evaluation completed successfully!")


if __name__ == '__main__':
    main()