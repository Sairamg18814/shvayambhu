#!/usr/bin/env python3
"""Test MLX utility wrapper classes."""

from utils import *
import mlx.core as mx
import mlx.nn as nn

def test_device_manager():
    """Test DeviceManager."""
    print("=== Testing DeviceManager ===")
    dm = get_device_manager()
    
    info = dm.get_device_info()
    print(f"Device info: {info}")
    
    dm.synchronize()
    print("✅ Device synchronized")
    
    dm.clear_cache()
    print("✅ Cache cleared\n")


def test_tensor_ops():
    """Test TensorOps."""
    print("=== Testing TensorOps ===")
    
    # Create tensors
    t1 = TensorOps.zeros((3, 3))
    print(f"Zeros: {t1}")
    
    t2 = TensorOps.ones((2, 2))
    print(f"Ones: {t2}")
    
    t3 = TensorOps.randn((2, 3), mean=0, std=0.1)
    print(f"Random normal: {t3}")
    
    # Test conversions
    np_array = TensorOps.to_numpy(t1)
    print(f"✅ Converted to numpy: shape={np_array.shape}")
    
    t4 = TensorOps.create_tensor([[1, 2], [3, 4]])
    print(f"✅ Created from list: {t4}\n")


def test_model_checkpoint():
    """Test ModelCheckpoint."""
    print("=== Testing ModelCheckpoint ===")
    
    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
    
    model = SimpleModel()
    checkpoint = ModelCheckpoint(save_dir="test_checkpoints")
    
    # Save
    metadata = {'epoch': 1, 'loss': 0.5}
    checkpoint.save(model, "test_model", metadata)
    print("✅ Model saved")
    
    # Create new model and load
    new_model = SimpleModel()
    loaded_meta = checkpoint.load(new_model, "test_model")
    print(f"✅ Model loaded, metadata: {loaded_meta}\n")


def test_gradient_utils():
    """Test GradientUtils."""
    print("=== Testing GradientUtils ===")
    
    # Simple model and loss
    model = nn.Linear(5, 2)
    
    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        return mx.mean((logits - targets) ** 2)
    
    # Test data
    inputs = mx.random.normal((10, 5))
    targets = mx.random.normal((10, 2))
    
    # Compute gradients
    loss, grads = GradientUtils.compute_gradient(loss_fn, model, inputs, targets)
    print(f"Loss: {loss:.4f}")
    print(f"✅ Gradients computed: {len(grads)} parameters")
    
    # Clip gradients
    clipped = GradientUtils.clip_gradients(grads, max_norm=1.0)
    print("✅ Gradients clipped\n")


def test_data_utils():
    """Test DataUtils."""
    print("=== Testing DataUtils ===")
    
    # Create batches
    data = mx.arange(100).reshape(100, 1)
    batches = DataUtils.create_batches(data, batch_size=32)
    print(f"Created {len(batches)} batches from 100 samples")
    print(f"✅ Batch shapes: {[b.shape for b in batches[:3]]}")
    
    # Pad sequences
    sequences = [
        mx.array([1, 2, 3]),
        mx.array([4, 5]),
        mx.array([6, 7, 8, 9])
    ]
    padded = DataUtils.pad_sequences(sequences)
    print(f"✅ Padded sequences: shape={padded.shape}\n")


def test_profiler():
    """Test Profiler."""
    print("=== Testing Profiler ===")
    
    profiler = Profiler()
    
    # Profile some operations
    for i in range(5):
        a = mx.random.normal((1000, 1000))
        b = mx.random.normal((1000, 1000))
        
        result = profiler.profile("matmul", mx.matmul, a, b)
        result = profiler.profile("add", mx.add, a, b)
    
    # Get report
    report = profiler.report()
    for op, stats in report.items():
        print(f"{op}: {stats['mean']*1000:.2f} ± {stats['std']*1000:.2f} ms")
    
    print("✅ Profiling complete\n")


def main():
    """Run all tests."""
    print("Testing MLX Utility Wrapper Classes\n")
    
    test_device_manager()
    test_tensor_ops()
    test_model_checkpoint()
    test_gradient_utils()
    test_data_utils()
    test_profiler()
    
    print("=== All Tests Passed ✅ ===")
    print("MLX utilities are ready for use!")


if __name__ == "__main__":
    main()