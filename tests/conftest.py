"""Pytest configuration and fixtures."""

import pytest
import torch
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "The quick brown fox jumps over the lazy dog."


@pytest.fixture
def sample_bytes():
    """Sample bytes for testing BLT."""
    return b"Hello, world! \xf0\x9f\x8c\x8d"  # Includes emoji


@pytest.fixture
def mock_model_config():
    """Mock configuration for model testing."""
    return {
        "hidden_dim": 512,
        "num_layers": 6,
        "num_heads": 8,
        "vocab_size": 256,  # Byte-level
        "max_seq_length": 1024,
        "dropout": 0.1,
    }