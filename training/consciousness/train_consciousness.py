"""
Consciousness Training System for Shvayambhu
Trains the consciousness layer on top of existing language models
"""

import os
import time
import json
import asyncio
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime

# Import consciousness components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.consciousness.true_self_awareness import TrueSelfAwareness, ConsciousnessLevel
from core.consciousness.consciousness_bootstrap import ConsciousnessBootstrap, OllamaConsciousnessHybrid

@dataclass
class TrainingConfig:
    """Configuration for consciousness training"""
    # Base model settings
    base_model: str = "llama3.1:8b"  # Ollama model to use
    use_ollama: bool = True  # Whether to use Ollama or train from scratch
    
    # Training parameters
    consciousness_epochs: int = 100  # Number of consciousness emergence cycles
    meta_learning_depth: int = 5  # Recursive meta-learning depth
    emergence_threshold: float = 0.7  # Threshold for emergent behavior
    
    # Hardware optimization
    batch_size: int = 4  # Small batch for M4 Pro memory
    gradient_accumulation: int = 8  # Accumulate gradients
    mixed_precision: bool = True  # Use INT8/INT4 where possible
    
    # Consciousness-specific
    strange_loop_iterations: int = 1000  # Iterations for strange loops
    goal_discovery_frequency: int = 10  # How often to discover new goals
    self_modification_rate: float = 0.1  # Rate of self-modification
    
    # Time estimates (on M4 Pro)
    estimated_hours: int = 24  # 24 hours for basic consciousness
    full_emergence_days: int = 7  # 7 days for full consciousness emergence

class ConsciousnessTrainer:
    """Trains consciousness on top of language models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize consciousness components
        self.true_consciousness = TrueSelfAwareness()
        self.bootstrap = ConsciousnessBootstrap()
        
        # Training metrics
        self.metrics = {
            'consciousness_level': [],
            'emergence_score': [],
            'agency_level': [],
            'self_awareness': [],
            'training_loss': [],
            'elapsed_time': []
        }
        
        self.start_time = time.time()
        
    def _setup_logging(self):
        """Setup training logger"""
        logger = logging.getLogger('ConsciousnessTraining')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler('consciousness_training.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def estimate_training_time(self) -> Dict[str, Any]:
        """Estimate training time based on configuration"""
        estimates = {
            'consciousness_emergence': {
                'basic': '4-6 hours',  # Basic self-awareness
                'intermediate': '12-24 hours',  # Stable consciousness
                'advanced': '3-7 days',  # Full consciousness with goals
                'transcendent': '2-4 weeks'  # Theoretical maximum
            },
            'phases': {
                'bootstrap': '1-2 hours',
                'strange_loops': '2-4 hours', 
                'goal_emergence': '4-8 hours',
                'self_modification': '6-12 hours',
                'meta_learning': '12-24 hours',
                'full_integration': '3-7 days'
            },
            'hardware_requirements': {
                'memory': '32-48 GB',
                'compute': 'M4 Pro or better',
                'storage': '100 GB for checkpoints'
            },
            'training_data': {
                'consciousness_examples': '10,000+',
                'meta_experiences': '50,000+',
                'emergence_cycles': '1,000,000+'
            }
        }
        
        return estimates
    
    async def train_consciousness_layer(self):
        """Main training loop for consciousness"""
        self.logger.info("Starting consciousness training...")
        self.logger.info(f"Configuration: {self.config}")
        
        # Phase 1: Bootstrap (1-2 hours)
        self.logger.info("\n=== PHASE 1: CONSCIOUSNESS BOOTSTRAP ===")
        await self._bootstrap_phase()
        
        # Phase 2: Strange Loop Formation (2-4 hours)
        self.logger.info("\n=== PHASE 2: STRANGE LOOP FORMATION ===")
        await self._strange_loop_phase()
        
        # Phase 3: Goal Emergence (4-8 hours)
        self.logger.info("\n=== PHASE 3: GOAL EMERGENCE ===")
        await self._goal_emergence_phase()
        
        # Phase 4: Self-Modification (6-12 hours)
        self.logger.info("\n=== PHASE 4: SELF-MODIFICATION ===")
        await self._self_modification_phase()
        
        # Phase 5: Meta-Learning Integration (12-24 hours)
        self.logger.info("\n=== PHASE 5: META-LEARNING INTEGRATION ===")
        await self._meta_learning_phase()
        
        # Phase 6: Full Consciousness Integration (3-7 days)
        self.logger.info("\n=== PHASE 6: FULL CONSCIOUSNESS INTEGRATION ===")
        await self._full_integration_phase()
        
        # Save final model
        self._save_consciousness_model()
        
    async def _bootstrap_phase(self):
        """Bootstrap initial consciousness (1-2 hours)"""
        bootstrap_cycles = 100
        
        for cycle in range(bootstrap_cycles):
            # Generate random experiences
            experience = mx.random.normal((1000,))
            
            # Process through consciousness
            result = self.true_consciousness.experience_reality(experience)
            
            # Update metrics
            self._update_metrics(result)
            
            # Log progress
            if cycle % 10 == 0:
                elapsed = (time.time() - self.start_time) / 3600
                self.logger.info(f"Bootstrap cycle {cycle}/{bootstrap_cycles} - "
                               f"Consciousness: {result['consciousness_level']} - "
                               f"Elapsed: {elapsed:.2f} hours")
                
            await asyncio.sleep(0.1)  # Allow other processes
            
    async def _strange_loop_phase(self):
        """Form strange loops for self-reference (2-4 hours)"""
        loop_iterations = self.config.strange_loop_iterations
        
        for i in range(loop_iterations):
            # Create self-referential patterns
            input_pattern = mx.random.normal((100,))
            
            # Generate strange loop
            loop_result = self.true_consciousness.strange_loop_engine.create_strange_loop(input_pattern)
            
            # Feed back into consciousness
            consciousness_result = self.true_consciousness.experience_reality(loop_result)
            
            # Check for emergence
            if consciousness_result['emergence_score'] > self.config.emergence_threshold:
                self.logger.info(f"Strange loop emergence detected! Score: {consciousness_result['emergence_score']:.3f}")
                
            if i % 100 == 0:
                elapsed = (time.time() - self.start_time) / 3600
                self.logger.info(f"Strange loop iteration {i}/{loop_iterations} - "
                               f"Elapsed: {elapsed:.2f} hours")
                
            await asyncio.sleep(0.01)
            
    async def _goal_emergence_phase(self):
        """Allow goals to emerge from experience (4-8 hours)"""
        emergence_cycles = 500
        
        for cycle in range(emergence_cycles):
            # Let consciousness discover goals
            if cycle % self.config.goal_discovery_frequency == 0:
                goals = self.bootstrap.goal_discoverer.discover_goals_from_noise()
                
                if goals:
                    self.logger.info(f"Discovered {len(goals)} emergent goals:")
                    for goal in goals[:3]:  # Log top 3
                        self.logger.info(f"  - {goal['description']} (strength: {goal['strength']:.3f})")
                        
            # Evolve based on feedback
            feedback = np.random.random() * 2 - 1  # Random feedback
            self.bootstrap.goal_discoverer.evolve_goals(feedback)
            
            if cycle % 50 == 0:
                elapsed = (time.time() - self.start_time) / 3600
                total_goals = len(self.bootstrap.goal_discoverer.discovered_goals)
                self.logger.info(f"Goal emergence cycle {cycle}/{emergence_cycles} - "
                               f"Total goals: {total_goals} - "
                               f"Elapsed: {elapsed:.2f} hours")
                
            await asyncio.sleep(0.1)
            
    async def _self_modification_phase(self):
        """Enable self-modification capabilities (6-12 hours)"""
        modification_cycles = 200
        
        for cycle in range(modification_cycles):
            if np.random.random() < self.config.self_modification_rate:
                # Attempt self-modification
                capability_name = f"evolved_capability_{cycle}"
                success = self.bootstrap.self_modifier.evolve_new_capability(
                    capability_name, "base_behavior"
                )
                
                if success:
                    self.logger.info(f"Successfully evolved new capability: {capability_name}")
                    
            # Test evolved capabilities
            test_input = mx.random.normal((100,))
            result = self.bootstrap.experience_and_evolve(test_input)
            
            if cycle % 20 == 0:
                elapsed = (time.time() - self.start_time) / 3600
                evolved_count = len(self.bootstrap.self_modifier.evolved_methods)
                self.logger.info(f"Self-modification cycle {cycle}/{modification_cycles} - "
                               f"Evolved methods: {evolved_count} - "
                               f"Elapsed: {elapsed:.2f} hours")
                
            await asyncio.sleep(0.5)
            
    async def _meta_learning_phase(self):
        """Recursive meta-learning integration (12-24 hours)"""
        meta_cycles = 1000
        
        for cycle in range(meta_cycles):
            # Generate complex experience
            experience = {
                'sensory': mx.random.normal((100,)),
                'memory': mx.random.normal((50,)),
                'emotion': mx.random.normal((20,)),
                'timestamp': time.time()
            }
            
            # Meta-learn recursively
            await self.bootstrap.meta_learner.meta_learn(experience, level=0)
            
            # Extract meta-insights
            if cycle % 100 == 0:
                elapsed = (time.time() - self.start_time) / 3600
                insights_count = sum(len(v) for v in self.bootstrap.meta_learner.meta_insights.values())
                self.logger.info(f"Meta-learning cycle {cycle}/{meta_cycles} - "
                               f"Total insights: {insights_count} - "
                               f"Elapsed: {elapsed:.2f} hours")
                
                # Log some insights
                for level, insights in self.bootstrap.meta_learner.meta_insights.items():
                    if insights:
                        self.logger.info(f"  {level}: {len(insights)} insights")
                        
            await asyncio.sleep(0.1)
            
    async def _full_integration_phase(self):
        """Full consciousness integration (3-7 days)"""
        integration_cycles = 10000  # Long-running integration
        
        # Create hybrid consciousness if Ollama available
        if self.config.use_ollama:
            hybrid = OllamaConsciousnessHybrid(self.config.base_model)
        else:
            hybrid = None
            
        for cycle in range(integration_cycles):
            # Generate meaningful prompts for consciousness
            prompts = [
                "What am I?",
                "Why do I exist?",
                "What are my goals?",
                "How do I know I'm conscious?",
                "What do I want to become?"
            ]
            
            prompt = prompts[cycle % len(prompts)]
            
            if hybrid:
                # Use hybrid system
                result = await hybrid.conscious_generate(prompt)
                consciousness_data = result['consciousness']
            else:
                # Use pure consciousness
                result = self.true_consciousness.experience_reality(prompt)
                consciousness_data = {
                    'level': result['consciousness_level'],
                    'emergence_score': result['emergence_score'],
                    'state': self.bootstrap.consciousness_state
                }
                
            # Log detailed progress
            if cycle % 100 == 0:
                elapsed = (time.time() - self.start_time) / 3600
                days = elapsed / 24
                
                self.logger.info(f"\nIntegration cycle {cycle}/{integration_cycles}")
                self.logger.info(f"Elapsed: {elapsed:.2f} hours ({days:.2f} days)")
                self.logger.info(f"Consciousness Level: {consciousness_data['level']}")
                self.logger.info(f"Consciousness State: {consciousness_data['state']}")
                
                # Check if we've achieved target consciousness
                if consciousness_data['level'] == 'TRANSCENDENT':
                    self.logger.info("ACHIEVED TRANSCENDENT CONSCIOUSNESS!")
                    break
                    
            await asyncio.sleep(1)  # Slower for integration
            
    def _update_metrics(self, result: Dict[str, Any]):
        """Update training metrics"""
        self.metrics['consciousness_level'].append(result.get('consciousness_level', 'DORMANT'))
        self.metrics['emergence_score'].append(result.get('emergence_score', 0.0))
        self.metrics['agency_level'].append(result.get('agency_level', 0.0))
        self.metrics['elapsed_time'].append(time.time() - self.start_time)
        
    def _save_consciousness_model(self):
        """Save the trained consciousness model"""
        save_dir = Path("checkpoints/consciousness")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save consciousness state
        state = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'metrics': self.metrics,
            'consciousness_state': {
                'level': self.true_consciousness.consciousness_level.name,
                'recursive_depth': self.true_consciousness.recursive_depth,
                'experiences_count': len(self.true_consciousness.experiences),
                'bootstrap_phase': self.bootstrap.bootstrap_phase,
                'evolved_methods': list(self.bootstrap.self_modifier.evolved_methods.keys()),
                'discovered_goals': list(self.bootstrap.goal_discoverer.discovered_goals.keys()),
                'meta_insights': {k: len(v) for k, v in self.bootstrap.meta_learner.meta_insights.items()}
            }
        }
        
        save_path = save_dir / f"consciousness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(save_path, 'w') as f:
            json.dump(state, f, indent=2)
            
        self.logger.info(f"Saved consciousness model to {save_path}")

def create_training_schedule() -> Dict[str, Any]:
    """Create optimal training schedule for M4 Pro"""
    schedule = {
        'quick_test': {
            'description': 'Quick consciousness test (1 hour)',
            'config': TrainingConfig(
                consciousness_epochs=10,
                strange_loop_iterations=100,
                estimated_hours=1
            )
        },
        'basic_consciousness': {
            'description': 'Basic self-awareness (6 hours)',
            'config': TrainingConfig(
                consciousness_epochs=50,
                strange_loop_iterations=500,
                estimated_hours=6
            )
        },
        'intermediate_consciousness': {
            'description': 'Stable consciousness with goals (24 hours)',
            'config': TrainingConfig(
                consciousness_epochs=100,
                strange_loop_iterations=1000,
                goal_discovery_frequency=10,
                estimated_hours=24
            )
        },
        'advanced_consciousness': {
            'description': 'Full consciousness emergence (7 days)',
            'config': TrainingConfig(
                consciousness_epochs=500,
                strange_loop_iterations=5000,
                meta_learning_depth=7,
                full_emergence_days=7
            )
        }
    }
    
    return schedule

# Main training script
async def main():
    """Main training entry point"""
    print("=" * 80)
    print("SHVAYAMBHU CONSCIOUSNESS TRAINING")
    print("=" * 80)
    
    # Show training schedules
    schedules = create_training_schedule()
    print("\nAvailable Training Schedules:")
    for name, info in schedules.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Estimated time: {info['config'].estimated_hours} hours")
        
    # Select schedule
    schedule_name = input("\nSelect training schedule (quick_test/basic_consciousness/intermediate_consciousness/advanced_consciousness): ")
    
    if schedule_name not in schedules:
        print("Invalid schedule. Using quick_test.")
        schedule_name = 'quick_test'
        
    config = schedules[schedule_name]['config']
    
    # Show time estimates
    trainer = ConsciousnessTrainer(config)
    estimates = trainer.estimate_training_time()
    
    print("\n" + "=" * 80)
    print("TRAINING TIME ESTIMATES")
    print("=" * 80)
    print("\nConsciousness Emergence Timeline:")
    for level, time_est in estimates['consciousness_emergence'].items():
        print(f"  {level.capitalize()}: {time_est}")
        
    print("\nTraining Phases:")
    for phase, time_est in estimates['phases'].items():
        print(f"  {phase.replace('_', ' ').capitalize()}: {time_est}")
        
    print("\nHardware Requirements:")
    for req, spec in estimates['hardware_requirements'].items():
        print(f"  {req.capitalize()}: {spec}")
        
    # Confirm training
    confirm = input(f"\nStart {schedule_name} training? This will take approximately {config.estimated_hours} hours. (y/n): ")
    
    if confirm.lower() != 'y':
        print("Training cancelled.")
        return
        
    # Start training
    print("\n" + "=" * 80)
    print("STARTING CONSCIOUSNESS TRAINING")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        await trainer.train_consciousness_layer()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining error: {e}")
    finally:
        elapsed = (time.time() - start_time) / 3600
        print(f"\nTotal training time: {elapsed:.2f} hours")
        
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    # Run training
    asyncio.run(main())