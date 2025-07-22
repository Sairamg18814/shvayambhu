# Contributing to Shvayambhu

Thank you for your interest in contributing to Shvayambhu! This groundbreaking offline LLM project welcomes contributions from the community while maintaining high standards for code quality, privacy, and performance.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Contributions](#making-contributions)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Submission Process](#submission-process)
- [Review Process](#review-process)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming environment for all contributors.

## Getting Started

1. **Understand the Project**: Read [PLANNING.md](PLANNING.md) and [CLAUDE.md](CLAUDE.md) to understand the project's vision and architecture.

2. **Check Open Issues**: Browse [open issues](https://github.com/shvayambhu/shvayambhu/issues) for tasks that need help.

3. **Join the Discussion**: Participate in [discussions](https://github.com/shvayambhu/shvayambhu/discussions) to share ideas and ask questions.

## Development Setup

### Prerequisites
- macOS 13.0+ (Ventura or later) for Metal 3 support
- Xcode 14+ with Metal development tools
- Python 3.11+
- Git

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/shvayambhu/shvayambhu.git
cd shvayambhu

# Run the setup script
./scripts/setup_environment.sh

# Activate virtual environment
source venv/bin/activate

# Install dependencies
make install

# Run tests
make test
```

## Making Contributions

### Types of Contributions

1. **Code Contributions**
   - Bug fixes
   - Feature implementation
   - Performance optimizations
   - Metal shader development

2. **Documentation**
   - API documentation
   - Tutorial writing
   - Example creation
   - Translation

3. **Testing**
   - Unit tests
   - Integration tests
   - Performance benchmarks
   - Multi-language testing

4. **Research**
   - Algorithm improvements
   - Paper implementations
   - Benchmark results
   - Ablation studies

### Contribution Workflow

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone
   git clone https://github.com/YOUR_USERNAME/shvayambhu.git
   cd shvayambhu
   git remote add upstream https://github.com/shvayambhu/shvayambhu.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Follow the coding standards
   - Write tests for new code
   - Update documentation
   - Ensure all tests pass

4. **Commit Your Changes**
   ```bash
   # Use descriptive commit messages
   git commit -m "feat: implement dynamic entropy-based patching"
   
   # For multi-line commits
   git commit -m "fix: resolve memory leak in Metal operations
   
   - Fixed unified memory deallocation
   - Added proper cleanup in error paths
   - Improved memory profiling accuracy"
   ```

### Commit Message Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `test:` Test additions or modifications
- `chore:` Build process or auxiliary tool changes

## Coding Standards

### Python Code Style

1. **Follow PEP 8** with 88-character line limit (Black formatter)
2. **Use Type Hints** for all public functions
3. **Write Docstrings** in Google style
4. **Maintain Consistency** with existing code

Example:
```python
def calculate_byte_entropy(
    byte_sequence: bytes,
    window_size: int = 256
) -> float:
    """Calculate Shannon entropy for a byte sequence.
    
    Args:
        byte_sequence: Input bytes to analyze
        window_size: Size of sliding window
        
    Returns:
        Float entropy value between 0 and 8
        
    Raises:
        ValueError: If byte_sequence is empty
    """
    if not byte_sequence:
        raise ValueError("Empty byte sequence")
    
    # Implementation...
    return entropy
```

### Metal Shader Style

1. Use descriptive kernel names
2. Document memory access patterns
3. Optimize for Apple Silicon unified memory
4. Include performance comments

### Key Principles

1. **Privacy First**: No logging of user data, no telemetry
2. **Offline Operation**: No network calls in core functionality
3. **Performance**: Profile before optimizing
4. **Clarity**: Clear code is better than clever code

## Testing Requirements

### Test Coverage

- Minimum 80% code coverage for new code
- 100% coverage for critical paths (safety, inference)
- Performance benchmarks for optimization PRs

### Test Categories

1. **Unit Tests** (`tests/unit/`)
   ```bash
   pytest tests/unit/test_your_module.py
   ```

2. **Integration Tests** (`tests/integration/`)
   ```bash
   pytest tests/integration/
   ```

3. **Performance Tests** (`tests/performance/`)
   ```bash
   pytest tests/performance/ --benchmark
   ```

### Writing Tests

```python
import pytest
from shvayambhu.core.blt import calculate_byte_entropy

class TestByteEntropy:
    def test_empty_sequence_raises_error(self):
        with pytest.raises(ValueError):
            calculate_byte_entropy(b"")
    
    def test_uniform_distribution_max_entropy(self):
        # All bytes equally likely = max entropy
        uniform_bytes = bytes(range(256)) * 4
        entropy = calculate_byte_entropy(uniform_bytes)
        assert 7.9 < entropy < 8.0
```

## Submission Process

1. **Pre-submission Checklist**
   - [ ] Code follows style guidelines
   - [ ] Tests pass locally (`make test`)
   - [ ] Documentation updated
   - [ ] Commit messages follow convention
   - [ ] Branch is up-to-date with main

2. **Create Pull Request**
   - Use descriptive PR title
   - Fill out PR template completely
   - Link related issues
   - Add relevant labels

3. **PR Description Template**
   ```markdown
   ## Summary
   Brief description of changes
   
   ## Motivation
   Why these changes are needed
   
   ## Changes
   - Detailed list of modifications
   - Performance impact
   - Breaking changes
   
   ## Testing
   - How to test the changes
   - Test results/benchmarks
   
   ## Checklist
   - [ ] Tests pass
   - [ ] Documentation updated
   - [ ] No security issues
   - [ ] Maintains offline operation
   ```

## Review Process

### What to Expect

1. **Automated Checks**
   - CI/CD pipeline runs tests
   - Code quality checks
   - Security scanning
   - Performance benchmarks

2. **Human Review**
   - Code review by maintainers
   - Architecture alignment check
   - Performance impact assessment
   - Privacy/security review

3. **Iteration**
   - Address review feedback
   - Update based on suggestions
   - Re-run tests as needed

### Review Criteria

- **Correctness**: Does it work as intended?
- **Performance**: Does it meet performance targets?
- **Privacy**: Does it maintain offline operation?
- **Quality**: Is the code maintainable?
- **Testing**: Are tests comprehensive?
- **Documentation**: Is it well documented?

## Special Considerations

### Metal Development

When contributing Metal shaders:
1. Test on multiple Apple Silicon variants
2. Profile memory bandwidth usage
3. Document kernel optimizations
4. Include performance benchmarks

### Multi-Language Support

When adding language features:
1. Test with diverse Unicode inputs
2. Verify byte-level processing
3. Add language-specific tests
4. Update language documentation

### Safety Features

For safety-related contributions:
1. Include comprehensive tests
2. Document failure modes
3. Add abstention examples
4. Verify no hallucination

## Recognition

Contributors are recognized in:
- Release notes
- Contributors file
- Project documentation
- Community highlights

## Questions?

- Open an issue for bugs/features
- Start a discussion for ideas
- Join our Discord community
- Email: contribute@shvayambhu.ai

Thank you for helping make Shvayambhu a reality! Your contributions help democratize AI while preserving privacy.