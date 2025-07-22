#!/usr/bin/env python3
"""
Interactive chat with Shvayambhu - Better input handling
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Run the shvayambhu chat
from shvayambhu import main

if __name__ == "__main__":
    # Ensure we have a proper terminal
    if not sys.stdin.isatty():
        print("Error: This script requires an interactive terminal")
        sys.exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)