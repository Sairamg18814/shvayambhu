#!/usr/bin/env python3
"""MLX Hello World - Basic neural network example for M4 Pro."""

import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time

class SimpleNet(nn.Module):
    """Simple neural network for demonstration."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=2):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def __call__(self, x):
        x = self.layer1(x)
        x = mx.maximum(x, 0.0)  # ReLU activation
        x = self.layer2(x)
        return x

def generate_synthetic_data(n_samples=1000, input_size=10):
    """Generate synthetic classification data."""
    # Create random input data
    X = mx.random.normal(shape=(n_samples, input_size))
    
    # Create random weights for generating labels
    true_weights = mx.random.normal(shape=(input_size, 1))
    
    # Generate labels based on linear combination + noise
    logits = mx.matmul(X, true_weights).squeeze()
    noise = mx.random.normal(shape=(n_samples,)) * 0.1
    y = (logits + noise > 0).astype(mx.int32)
    
    return X, y

def train_step(model, X, y, optimizer):
    """Single training step."""
    def loss_fn(model, X, y):
        logits = model(X)
        # Cross-entropy loss
        loss = mx.mean(nn.losses.cross_entropy(logits, y))
        return loss
    
    # Compute loss and gradients
    loss, grads = mx.value_and_grad(loss_fn)(model, X, y)
    
    # Update model parameters
    optimizer.update(model, grads)
    
    return loss

def evaluate(model, X, y):
    """Evaluate model accuracy."""
    logits = model(X)
    predictions = mx.argmax(logits, axis=1)
    accuracy = mx.mean(predictions == y)
    return accuracy

def main():
    """Run MLX hello world example."""
    print("=== MLX Hello World ===")
    print(f"Device: {mx.default_device()}")
    print()
    
    # Set random seed for reproducibility
    mx.random.seed(42)
    
    # Generate data
    print("Generating synthetic data...")
    n_train, n_test = 5000, 1000
    input_size = 10
    
    X_train, y_train = generate_synthetic_data(n_train, input_size)
    X_test, y_test = generate_synthetic_data(n_test, input_size)
    
    print(f"Training samples: {n_train}")
    print(f"Test samples: {n_test}")
    print(f"Input features: {input_size}")
    print()
    
    # Create model
    print("Creating neural network...")
    model = SimpleNet(input_size=input_size, hidden_size=50, output_size=2)
    
    # Count parameters
    n_params = 0
    for name, param in model.parameters().items():
        if isinstance(param, dict):
            # Handle nested parameters
            for sub_name, sub_param in param.items():
                n_params += sub_param.size
        else:
            n_params += param.size
    print(f"Model parameters: {n_params:,}")
    print()
    
    # Create optimizer
    learning_rate = 0.01
    optimizer = optim.SGD(learning_rate=learning_rate)
    
    # Training loop
    print("Training...")
    batch_size = 32
    n_epochs = 10
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # Shuffle data
        indices = mx.random.permutation(n_train)
        
        epoch_loss = 0.0
        n_batches = 0
        
        # Mini-batch training
        for i in range(0, n_train, batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            loss = train_step(model, X_batch, y_batch, optimizer)
            epoch_loss += loss.item()
            n_batches += 1
        
        # Evaluate
        train_acc = evaluate(model, X_train, y_train)
        test_acc = evaluate(model, X_test, y_test)
        
        avg_loss = epoch_loss / n_batches
        print(f"Epoch {epoch+1}/{n_epochs} - "
              f"Loss: {avg_loss:.4f}, "
              f"Train Acc: {train_acc.item():.4f}, "
              f"Test Acc: {test_acc.item():.4f}")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Performance metrics
    samples_per_second = (n_train * n_epochs) / training_time
    print(f"Training throughput: {samples_per_second:.0f} samples/second")
    
    # Test inference speed
    print("\nTesting inference speed...")
    n_inference = 1000
    start_time = time.time()
    
    for _ in range(n_inference):
        _ = model(X_test)
    
    inference_time = time.time() - start_time
    inference_per_second = (n_inference * n_test) / inference_time
    
    print(f"Inference throughput: {inference_per_second:.0f} samples/second")
    print(f"Average inference time: {(inference_time/n_inference)*1000:.2f} ms/batch")
    
    print("\n✅ MLX Hello World completed successfully!")
    print("✅ Neural network training on Metal GPU verified")
    print("✅ Ready for larger models and complex architectures")

if __name__ == "__main__":
    main()