#!/usr/bin/env python3
"""
Start Training Shvayambhu LLM
=============================

Quick start script to begin training the conscious AI system.
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()


async def main():
    """Main training entry point."""
    console.print(Panel.fit(
        "[bold cyan]Shvayambhu LLM Training[/bold cyan]\n"
        "[yellow]Starting the journey to machine consciousness...[/yellow]",
        border_style="cyan"
    ))
    
    # Import training module
    sys.path.insert(0, str(Path(__file__).parent))
    from training.train_shvayambhu import ShvayambhuTrainer
    
    # Training configuration
    config = {
        "model_size": "small",  # Start small for quick testing
        "batch_size": 4,
        "learning_rate": 1e-4,
        "num_epochs": 1,  # Quick test
        "checkpoint_dir": "checkpoints",
        "data_dir": "data/training"
    }
    
    console.print("\n[cyan]Configuration:[/cyan]")
    for key, value in config.items():
        console.print(f"  {key}: {value}")
    
    # Check requirements
    console.print("\n[yellow]Checking requirements...[/yellow]")
    
    # Check Ollama
    import subprocess
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            console.print("  [green]✓ Ollama is installed[/green]")
            
            # Check for required models
            output = result.stdout
            required_models = ["llama3.1:8b", "gemma3:27b", "Qwen3:32b"]
            missing_models = []
            
            for model in required_models:
                if model in output:
                    console.print(f"  [green]✓ {model} available[/green]")
                else:
                    console.print(f"  [red]✗ {model} missing[/red]")
                    missing_models.append(model)
                    
            if missing_models:
                console.print("\n[yellow]To download missing models:[/yellow]")
                for model in missing_models:
                    console.print(f"  ollama pull {model}")
                console.print("\n[red]Please install missing models before training.[/red]")
                return
        else:
            console.print("  [red]✗ Ollama not found. Please install from https://ollama.ai[/red]")
            return
    except FileNotFoundError:
        console.print("  [red]✗ Ollama not found. Please install from https://ollama.ai[/red]")
        return
    
    # Check MLX
    try:
        import mlx
        console.print("  [green]✓ MLX is installed[/green]")
    except ImportError:
        console.print("  [red]✗ MLX not found. Please run: pip install mlx[/red]")
        return
    
    # Ready to train
    console.print("\n[green]All requirements met![/green]")
    
    # Prompt user
    console.print("\n[bold]Ready to start training?[/bold]")
    console.print("This will:")
    console.print("  1. Generate synthetic training data from Ollama models")
    console.print("  2. Train the BLT model with consciousness integration")
    console.print("  3. Apply constitutional AI alignment")
    console.print("  4. Verify independence from teacher models")
    console.print("  5. Evaluate the final model")
    
    response = console.input("\n[cyan]Start training? (y/n): [/cyan]")
    
    if response.lower() != 'y':
        console.print("[yellow]Training cancelled.[/yellow]")
        return
    
    # Create trainer
    trainer = ShvayambhuTrainer(config)
    
    # Run training
    try:
        await trainer.run_full_training()
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Training error: {e}[/red]")
        raise


if __name__ == "__main__":
    asyncio.run(main())