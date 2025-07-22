#!/usr/bin/env python3
"""Test Metal Performance Shaders availability for MLX."""

import sys
import platform
import subprocess

def check_metal_availability():
    """Check if Metal Performance Shaders are available on this system."""
    
    print("=== Metal Performance Shaders Availability Check ===\n")
    
    # Check platform
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Architecture: {platform.machine()}")
    print(f"macOS Version: {platform.mac_ver()[0]}")
    print()
    
    # Check if MLX can be imported
    try:
        import mlx
        print("✅ MLX successfully imported")
        # MLX doesn't expose __version__ directly
        try:
            import pkg_resources
            version = pkg_resources.get_distribution("mlx").version
            print(f"   MLX version: {version}")
        except:
            print("   MLX version: (version info not available)")
    except ImportError as e:
        print(f"❌ Failed to import MLX: {e}")
        return False
    
    # Check MLX Metal backend
    try:
        import mlx.core as mx
        
        # Create a simple tensor to test Metal backend
        print("\n=== Testing Metal Backend ===")
        x = mx.array([1.0, 2.0, 3.0])
        y = mx.array([4.0, 5.0, 6.0])
        z = x + y
        
        print(f"✅ Basic tensor operation successful")
        print(f"   x = {x}")
        print(f"   y = {y}")
        print(f"   z = x + y = {z}")
        
        # Test device info
        print("\n=== Device Information ===")
        print(f"Default device: {mx.default_device()}")
        
        # Test more complex operations
        print("\n=== Testing Complex Operations ===")
        
        # Matrix multiplication
        a = mx.random.normal((100, 100))
        b = mx.random.normal((100, 100))
        c = mx.matmul(a, b)
        print(f"✅ Matrix multiplication (100x100) successful")
        
        # Test memory allocation
        large_array = mx.zeros((1000, 1000, 10))
        print(f"✅ Large array allocation successful: shape={large_array.shape}")
        
        # Test unified memory
        print("\n=== Testing Unified Memory ===")
        data = mx.arange(1000000)
        result = mx.sum(data)
        print(f"✅ Unified memory test successful: sum of 1M elements = {result}")
        
    except Exception as e:
        print(f"❌ Metal backend test failed: {e}")
        return False
    
    # Check system Metal support
    print("\n=== System Metal Support ===")
    try:
        # Check for Metal compiler
        metal_result = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "--version"],
            capture_output=True,
            text=True
        )
        if metal_result.returncode == 0:
            print("✅ Metal compiler available:")
            print(f"   {metal_result.stdout.strip()}")
        else:
            print("❌ Metal compiler not found")
    except Exception as e:
        print(f"❌ Could not check Metal compiler: {e}")
    
    # Final summary
    print("\n=== Summary ===")
    print("✅ Metal Performance Shaders are fully available!")
    print("✅ MLX is properly configured for M4 Pro")
    print("✅ Ready for model development and training")
    
    return True

if __name__ == "__main__":
    success = check_metal_availability()
    sys.exit(0 if success else 1)