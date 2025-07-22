#!/usr/bin/env python3
"""
Shvayambhu Quick Start
=====================

The easiest way to use Shvayambhu LLM - no setup required!

Just run:
    python quickstart.py
"""

import subprocess
import sys
import os
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    required = {
        'mlx': 'mlx',
        'rich': 'rich',
        'numpy': 'numpy',
        'ollama': None  # Binary, not pip
    }
    
    missing = []
    for module, package in required.items():
        if package:  # Python package
            try:
                __import__(module)
            except ImportError:
                missing.append(package)
    
    # Check ollama
    try:
        subprocess.run(['ollama', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Ollama not found. Please install from https://ollama.ai")
        print("   On macOS: brew install ollama")
        
    return missing


def install_dependencies(packages):
    """Install missing dependencies."""
    if packages:
        print(f"📦 Installing dependencies: {', '.join(packages)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
        print("✅ Dependencies installed!")


def download_models():
    """Check and download required models."""
    print("\n🤖 Checking models...")
    
    # Check if models exist
    models_needed = []
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        installed_models = result.stdout.lower()
        
        if 'llama3.1:8b' not in installed_models:
            models_needed.append('llama3.1:8b')
            
    except:
        print("⚠️  Could not check Ollama models")
        return
        
    # Download if needed
    if models_needed:
        print(f"📥 Downloading models: {', '.join(models_needed)}")
        print("   This may take a few minutes...")
        for model in models_needed:
            subprocess.run(['ollama', 'pull', model])
        print("✅ Models ready!")
    else:
        print("✅ All models already installed!")


def create_simple_config():
    """Create a simple configuration file."""
    config_path = Path.home() / '.shvayambhu' / 'config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not config_path.exists():
        import json
        config = {
            "model_size": "medium",
            "temperature": 0.8,
            "features": {
                "consciousness": True,
                "memory": True,
                "emotional": True,
                "safety": True
            }
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Created config at {config_path}")


def main():
    """Quick start Shvayambhu."""
    print("🚀 Shvayambhu Quick Start")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
        
    # Check dependencies
    print("\n📋 Checking dependencies...")
    missing = check_dependencies()
    if missing:
        install_dependencies(missing)
        
    # Download models if needed
    download_models()
    
    # Create config
    create_simple_config()
    
    # Run Shvayambhu
    print("\n" + "=" * 50)
    print("✨ Starting Shvayambhu...")
    print("=" * 50 + "\n")
    
    # Import and run
    try:
        from shvayambhu import main as shvayambhu_main
        shvayambhu_main()
    except ImportError:
        # Fallback to running as script
        script_path = Path(__file__).parent / 'shvayambhu.py'
        subprocess.run([sys.executable, str(script_path)])


if __name__ == "__main__":
    main()