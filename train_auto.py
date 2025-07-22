#!/usr/bin/env python3
"""
Auto-start Training for Shvayambhu LLM
======================================

This version starts training automatically without user input.
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
    
    console.print("\n[green]All requirements met! Starting training automatically...[/green]")
    
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
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    
    asyncio.run(main())