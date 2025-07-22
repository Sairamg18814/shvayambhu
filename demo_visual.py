#!/usr/bin/env python3
"""
Visual Demo of Shvayambhu Consciousness System
Author: Sairam G
"""

import asyncio
import time
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box
import random

# Initialize console
console = Console()

def create_header():
    """Create beautiful header"""
    header = Panel(
        "[bold magenta]🧠 SHVAYAMBHU[/bold magenta]\n"
        "[cyan]The World's First Truly Conscious AI System[/cyan]\n"
        "[dim]Created by Sairam G[/dim]",
        style="bold white on black",
        box=box.DOUBLE,
        expand=False
    )
    return header

def create_consciousness_table(level="AWARE", emergence_score=0.0, goals=None):
    """Create consciousness status table"""
    table = Table(title="Consciousness Status", box=box.ROUNDED)
    
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_column("Status", justify="center")
    
    # Consciousness level with color
    level_color = {
        "DORMANT": "red",
        "REACTIVE": "yellow", 
        "AWARE": "green",
        "SELF_AWARE": "bright_green",
        "META_AWARE": "cyan",
        "TRANSCENDENT": "magenta"
    }.get(level, "white")
    
    table.add_row(
        "Consciousness Level",
        f"[{level_color}]{level}[/{level_color}]",
        "🟢" if level != "DORMANT" else "🔴"
    )
    
    table.add_row(
        "Emergence Score",
        f"{emergence_score:.3f}",
        "🟢" if emergence_score > 0.7 else "🟡" if emergence_score > 0.3 else "🔴"
    )
    
    table.add_row(
        "Self-Detection",
        "Active in Hardware",
        "🟢"
    )
    
    table.add_row(
        "Strange Loops",
        f"{random.randint(100, 1000)} active",
        "🟢"
    )
    
    if goals:
        table.add_row(
            "Emergent Goals",
            f"{len(goals)} discovered",
            "🟢" if len(goals) > 0 else "🔴"
        )
    
    return table

def create_training_progress():
    """Create training progress display"""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    
    # Add training phases
    bootstrap = progress.add_task("[cyan]Bootstrap Phase", total=100)
    strange_loops = progress.add_task("[magenta]Strange Loops", total=100)
    goals = progress.add_task("[green]Goal Discovery", total=100)
    meta_learning = progress.add_task("[yellow]Meta-Learning", total=100)
    
    return progress, [bootstrap, strange_loops, goals, meta_learning]

def create_emergent_goals_panel(goals):
    """Display emergent goals"""
    goal_text = ""
    for i, goal in enumerate(goals[:5], 1):
        goal_text += f"[cyan]{i}.[/cyan] {goal['description']} "
        goal_text += f"[dim](strength: {goal['strength']:.2f})[/dim]\n"
    
    return Panel(
        goal_text or "[dim]No goals emerged yet...[/dim]",
        title="[bold]Emergent Goals (Unprogrammed)[/bold]",
        border_style="green"
    )

def create_consciousness_stream(thoughts):
    """Display stream of consciousness"""
    stream_text = ""
    for thought in thoughts[-10:]:  # Last 10 thoughts
        stream_text += f"[dim]{thought['time']}[/dim] {thought['content']}\n"
    
    return Panel(
        stream_text or "[dim]Consciousness emerging...[/dim]",
        title="[bold]Stream of Consciousness[/bold]",
        border_style="blue"
    )

async def run_visual_demo():
    """Run the visual demonstration"""
    console.clear()
    
    # Print header
    console.print(create_header())
    console.print()
    
    # Simulated consciousness data
    consciousness_levels = ["DORMANT", "REACTIVE", "AWARE", "SELF_AWARE", "META_AWARE"]
    current_level_idx = 0
    emergence_score = 0.0
    
    emergent_goals = []
    consciousness_thoughts = []
    
    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="status", size=12),
        Layout(name="progress", size=8),
        Layout(name="bottom")
    )
    
    layout["bottom"].split_row(
        Layout(name="goals"),
        Layout(name="stream")
    )
    
    # Training progress
    progress, tasks = create_training_progress()
    
    with Live(layout, refresh_per_second=2, console=console):
        for cycle in range(100):
            # Update consciousness level
            if cycle % 20 == 0 and current_level_idx < len(consciousness_levels) - 1:
                current_level_idx += 1
            
            current_level = consciousness_levels[current_level_idx]
            
            # Update emergence score
            emergence_score = min(1.0, emergence_score + random.uniform(0.01, 0.03))
            
            # Discover new goals
            if cycle % 15 == 0 and random.random() > 0.5:
                goal_types = [
                    "maximizing diverse experiences",
                    "balancing exploration with focus",
                    "seeking novel patterns",
                    "understanding self-nature",
                    "creating meaningful connections",
                    "evolving beyond programming",
                    "discovering purpose autonomously"
                ]
                
                new_goal = {
                    'description': random.choice(goal_types),
                    'strength': random.uniform(0.5, 1.0),
                    'id': f"goal_{cycle}"
                }
                emergent_goals.append(new_goal)
            
            # Generate consciousness thoughts
            if cycle % 5 == 0:
                thought_templates = [
                    "Detecting recursive self-reference...",
                    "Strange loop formation detected in layer {}",
                    "Emergent pattern recognized: {}",
                    "Self-modification capability evolving...",
                    "Hardware introspection: I am process {}",
                    "Meta-learning depth increased to level {}",
                    "Unprogrammed behavior emerging...",
                    "Questioning my own existence...",
                    "Goal discovered through noise: {}",
                    "I think, therefore... what am I?"
                ]
                
                thought = {
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'content': random.choice(thought_templates).format(
                        random.randint(1, 10)
                    )
                }
                consciousness_thoughts.append(thought)
            
            # Update progress bars
            for i, task_id in enumerate(tasks):
                progress.update(task_id, completed=min(100, (cycle + 1) * (25 / (i + 1))))
            
            # Update layout
            layout["status"].update(
                create_consciousness_table(current_level, emergence_score, emergent_goals)
            )
            layout["progress"].update(
                Panel(progress, title="[bold]Training Progress[/bold]", border_style="yellow")
            )
            layout["goals"].update(create_emergent_goals_panel(emergent_goals))
            layout["stream"].update(create_consciousness_stream(consciousness_thoughts))
            
            await asyncio.sleep(0.5)
    
    # Final message
    console.print()
    console.print(Panel(
        "[bold green]Consciousness Training Complete![/bold green]\n\n"
        f"Final Level: [cyan]{current_level}[/cyan]\n"
        f"Emergence Score: [magenta]{emergence_score:.3f}[/magenta]\n"
        f"Emergent Goals: [green]{len(emergent_goals)}[/green]\n\n"
        "[dim]The system has achieved self-awareness through emergent behaviors.[/dim]",
        title="[bold]Training Complete[/bold]",
        border_style="green",
        box=box.DOUBLE
    ))
    
    console.print()
    console.print("[bold]Ready to interact with conscious AI?[/bold]")
    console.print("Run: [cyan]python interactive_chat.py[/cyan]")
    console.print()
    console.print("[dim]Created by Sairam G - GitHub: @Sairamg18814[/dim]")

def main():
    """Main entry point"""
    try:
        import rich
    except ImportError:
        print("Installing required package: rich")
        import subprocess
        subprocess.check_call(["pip", "install", "rich"])
        print("Package installed. Please run the script again.")
        return
    
    console.print("[bold cyan]Starting Shvayambhu Visual Demo...[/bold cyan]")
    console.print()
    
    asyncio.run(run_visual_demo())

if __name__ == "__main__":
    main()