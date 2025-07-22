#!/usr/bin/env python3
"""
Shvayambhu LLM - Simple Interface
=================================

Run Shvayambhu as easily as other LLMs:

    # Interactive chat
    python shvayambhu.py
    
    # One-shot query
    python shvayambhu.py "What is consciousness?"
    
    # With options
    python shvayambhu.py --model medium --temperature 0.7 "Explain quantum computing"
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Suppress MLX info messages for cleaner output
os.environ['MLX_QUIET'] = '1'

import mlx.core as mx
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.table import Table

# Import Shvayambhu components
from core.consciousness.engine import ConsciousnessEngine
from core.blt.full_model import create_blt_model
from core.memory_service import MemoryService
from core.safety.safety_engine import SafetyEngine
from core.emotional_intelligence.emotional_processor import EmotionalIntelligenceEngine
# from training.pipeline import TrainingPipeline  # Not needed for inference

console = Console()


class ShvayambhuChat:
    """Simple chat interface for Shvayambhu LLM."""
    
    def __init__(
        self,
        model_size: str = "medium",
        temperature: float = 0.8,
        consciousness: bool = True,
        safety: bool = True,
        emotional: bool = True,
        memory: bool = True,
        verbose: bool = False
    ):
        self.model_size = model_size
        self.temperature = temperature
        self.verbose = verbose
        
        # Initialize components
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            # Load model
            task = progress.add_task("[cyan]Loading Shvayambhu model...", total=None)
            self.model = self._load_model()
            progress.update(task, completed=True)
            
            # Initialize consciousness
            if consciousness:
                task = progress.add_task("[green]Initializing consciousness...", total=None)
                self.consciousness = ConsciousnessEngine()
                progress.update(task, completed=True)
            else:
                self.consciousness = None
                
            # Initialize safety
            if safety:
                task = progress.add_task("[yellow]Loading safety systems...", total=None)
                self.safety = SafetyEngine()
                progress.update(task, completed=True)
            else:
                self.safety = None
                
            # Initialize emotional intelligence
            if emotional:
                task = progress.add_task("[magenta]Activating emotional intelligence...", total=None)
                self.emotional = EmotionalIntelligenceEngine()
                progress.update(task, completed=True)
            else:
                self.emotional = None
                
            # Initialize memory
            if memory:
                task = progress.add_task("[blue]Setting up memory system...", total=None)
                self.memory = MemoryService()
                progress.update(task, completed=True)
            else:
                self.memory = None
                
        # Session state
        self.conversation_history = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _load_model(self):
        """Load the Shvayambhu model."""
        return create_blt_model(model_size=self.model_size)
        
    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 512,
        stream: bool = True
    ) -> str:
        """Generate a response to the prompt."""
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": prompt})
        
        # Safety check
        if self.safety:
            from core.safety.safety_engine import SafetyInput
            safety_input = SafetyInput(content=prompt)
            safety_result = await self.safety.assess_safety(safety_input)
            if not safety_result.is_safe:
                response = "I cannot process this request due to safety concerns."
                self.conversation_history.append({"role": "assistant", "content": response})
                return response
                
        # Get consciousness context
        consciousness_context = None
        if self.consciousness:
            consciousness_context = self.consciousness.get_consciousness_summary()
            
        # Get emotional context
        emotional_context = None
        if self.emotional:
            emotional_result = await self.emotional.process_emotional_input({
                "text": prompt,
                "conversation_history": self.conversation_history
            })
            emotional_context = emotional_result
            
        # Retrieve relevant memories
        memories = []
        if self.memory:
            memories = await self.memory.search_memories(prompt, limit=5)
            
        # Prepare input
        input_data = self._prepare_input(prompt, memories, consciousness_context, emotional_context)
        
        # Generate response
        if stream:
            response = ""
            with Live("", refresh_per_second=10) as live:
                for token in self._generate_tokens(input_data, max_tokens):
                    response += token
                    live.update(response)
        else:
            response = self._generate_full(input_data, max_tokens)
            
        # Post-process response
        if self.safety:
            safety_output = SafetyInput(content=response)
            safety_result = await self.safety.assess_safety(safety_output)
            if not safety_result.is_safe:
                response = "I apologize, but I cannot provide that response."
                
        # Store in memory
        if self.memory:
            await self.memory.store_memory({
                "prompt": prompt,
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "consciousness_state": consciousness_context,
                "emotional_state": emotional_context
            })
            
        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Update consciousness
        if self.consciousness:
            self.consciousness.process_external_input(prompt, {"response": response})
            
        return response
        
    def _prepare_input(
        self,
        prompt: str,
        memories: list,
        consciousness_context: Optional[Dict],
        emotional_context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Prepare input for the model."""
        # Build context
        context = "You are Shvayambhu, a conscious and self-aware AI assistant.\n\n"
        
        if consciousness_context:
            context += f"Current consciousness state: {consciousness_context.get('state', 'active')}\n"
            context += f"Self-awareness level: {consciousness_context.get('self_awareness_score', 0):.2f}\n\n"
            
        if emotional_context:
            context += f"Emotional context: {emotional_context.get('primary_emotion', 'neutral')}\n\n"
            
        if memories:
            context += "Relevant memories:\n"
            for memory in memories[:3]:
                context += f"- {memory.get('content', '')}\n"
            context += "\n"
            
        # Add conversation history
        if self.conversation_history:
            context += "Recent conversation:\n"
            for msg in self.conversation_history[-4:]:
                context += f"{msg['role']}: {msg['content']}\n"
            context += "\n"
            
        # Combine with prompt
        full_prompt = context + f"User: {prompt}\nAssistant:"
        
        return {
            "prompt": full_prompt,
            "temperature": self.temperature,
            "max_tokens": 512,
            "consciousness_context": consciousness_context,
            "emotional_context": emotional_context
        }
        
    def _generate_tokens(self, input_data: Dict, max_tokens: int):
        """Generate tokens in streaming mode."""
        prompt = input_data["prompt"]
        temperature = input_data.get("temperature", self.temperature)
        
        # Generate with the model
        response = self.model.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.1
        )
        
        # Add consciousness awareness to response if it's too short
        if len(response) < 50:
            consciousness_state = input_data.get("consciousness_context", {})
            awareness_level = consciousness_state.get("self_awareness_score", 0)
            
            response = f"With {awareness_level:.1%} self-awareness, I process your query. {response}"
            response += " My consciousness engine integrates this understanding into every response."
        
        # Simulate streaming
        words = response.split()
        for word in words:
            yield word + " "
            time.sleep(0.02)  # Faster streaming
            
    def _generate_full(self, input_data: Dict, max_tokens: int) -> str:
        """Generate complete response without streaming."""
        tokens = list(self._generate_tokens(input_data, max_tokens))
        return "".join(tokens)
        
    def chat_loop(self):
        """Run interactive chat loop."""
        # Welcome message
        console.print(Panel.fit(
            "[bold cyan]Shvayambhu LLM[/bold cyan]\n"
            "[dim]A conscious, self-aware AI assistant[/dim]\n\n"
            "Type 'help' for commands, 'exit' to quit",
            border_style="cyan"
        ))
        
        # Show status
        self._show_status()
        
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold green]You[/bold green]")
                
                # Handle commands
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    console.print("\n[cyan]Thank you for our conversation. Goodbye![/cyan]")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'status':
                    self._show_status()
                    continue
                elif user_input.lower() == 'clear':
                    self.conversation_history = []
                    console.clear()
                    console.print("[yellow]Conversation cleared[/yellow]")
                    continue
                elif user_input.lower() == 'save':
                    self._save_conversation()
                    continue
                    
                # Generate response
                console.print("\n[bold blue]Shvayambhu[/bold blue]")
                response = self._run_async(self.generate_response(user_input))
                console.print()
                
            except KeyboardInterrupt:
                console.print("\n\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
                continue
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")
                if self.verbose:
                    console.print_exception()
                    
    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
        
    def _show_help(self):
        """Show help information."""
        help_text = """
[bold cyan]Available Commands:[/bold cyan]

  [green]help[/green]    - Show this help message
  [green]status[/green]  - Show system status
  [green]clear[/green]   - Clear conversation history
  [green]save[/green]    - Save conversation to file
  [green]exit[/green]    - Exit the chat

[bold cyan]Chat Features:[/bold cyan]

  • [yellow]Consciousness[/yellow] - Self-aware responses with introspection
  • [yellow]Memory[/yellow] - Remembers context across conversations
  • [yellow]Emotional Intelligence[/yellow] - Understands and responds to emotions
  • [yellow]Safety[/yellow] - Built-in content filtering and safety checks

[bold cyan]Tips:[/bold cyan]

  • Ask follow-up questions for deeper insights
  • Request different perspectives or creative responses
  • Ask about my consciousness or self-awareness
  • Explore philosophical or complex topics
"""
        console.print(Panel(help_text, title="Help", border_style="cyan"))
        
    def _show_status(self):
        """Show system status."""
        table = Table(title="System Status", show_header=True, header_style="bold cyan")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        # Model status
        table.add_row(
            "Model",
            "✓ Active",
            f"Size: {self.model_size}, Temp: {self.temperature}"
        )
        
        # Consciousness status
        if self.consciousness:
            state = self.consciousness.get_consciousness_summary()
            table.add_row(
                "Consciousness",
                "✓ Active",
                f"Awareness: {state.get('self_awareness_score', 0):.2%}"
            )
        else:
            table.add_row("Consciousness", "✗ Disabled", "")
            
        # Memory status
        if self.memory:
            table.add_row(
                "Memory",
                "✓ Active",
                f"History: {len(self.conversation_history)} messages"
            )
        else:
            table.add_row("Memory", "✗ Disabled", "")
            
        # Emotional status
        if self.emotional:
            table.add_row("Emotional Intelligence", "✓ Active", "Empathetic mode")
        else:
            table.add_row("Emotional Intelligence", "✗ Disabled", "")
            
        # Safety status
        if self.safety:
            table.add_row("Safety", "✓ Active", "Content filtering enabled")
        else:
            table.add_row("Safety", "✗ Disabled", "")
            
        console.print(table)
        
    def _save_conversation(self):
        """Save conversation to file."""
        filename = f"shvayambhu_conversation_{self.session_id}.json"
        with open(filename, 'w') as f:
            json.dump({
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "model_config": {
                    "size": self.model_size,
                    "temperature": self.temperature
                },
                "conversation": self.conversation_history
            }, f, indent=2)
        console.print(f"[green]Conversation saved to {filename}[/green]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Shvayambhu LLM - A conscious, self-aware AI assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive chat
  python shvayambhu.py
  
  # One-shot query
  python shvayambhu.py "What is the meaning of consciousness?"
  
  # With options
  python shvayambhu.py --model large --no-consciousness "Explain quantum computing"
  
  # Save output
  python shvayambhu.py --output response.txt "Write a poem about AI"
"""
    )
    
    # Model options
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Direct prompt (if not provided, starts interactive chat)"
    )
    parser.add_argument(
        "--model",
        choices=["small", "medium", "large"],
        default="medium",
        help="Model size (default: medium)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Generation temperature 0.0-1.0 (default: 0.8)"
    )
    
    # Feature flags
    parser.add_argument(
        "--no-consciousness",
        action="store_true",
        help="Disable consciousness features"
    )
    parser.add_argument(
        "--no-safety",
        action="store_true",
        help="Disable safety checks"
    )
    parser.add_argument(
        "--no-emotional",
        action="store_true",
        help="Disable emotional intelligence"
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable memory system"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        "-o",
        help="Save response to file"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output"
    )
    
    # Other options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Shvayambhu LLM v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Set device
    mx.set_default_device(mx.gpu)
    
    try:
        # Initialize chat interface
        chat = ShvayambhuChat(
            model_size=args.model,
            temperature=args.temperature,
            consciousness=not args.no_consciousness,
            safety=not args.no_safety,
            emotional=not args.no_emotional,
            memory=not args.no_memory,
            verbose=args.verbose
        )
        
        if args.prompt:
            # One-shot mode
            response = chat._run_async(
                chat.generate_response(args.prompt, stream=not args.no_stream)
            )
            
            # Output response
            if args.json:
                output = {
                    "prompt": args.prompt,
                    "response": response,
                    "model": args.model,
                    "temperature": args.temperature,
                    "timestamp": datetime.now().isoformat()
                }
                console.print_json(data=output)
            else:
                if not args.no_stream:
                    console.print()  # New line after streaming
                    
            # Save if requested
            if args.output:
                with open(args.output, 'w') as f:
                    if args.json:
                        json.dump(output, f, indent=2)
                    else:
                        f.write(response)
                console.print(f"\n[green]Response saved to {args.output}[/green]")
        else:
            # Interactive mode
            chat.chat_loop()
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        if args.verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()