# Contributing to GFRAM

Thank you for your interest in contributing to GFRAM! 🎉

We welcome contributions from the community and are grateful for any help you can provide, whether it's fixing bugs, adding new features, improving documentation, or suggesting ideas.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Guidelines](#development-guidelines)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Code of Conduct](#code-of-conduct)

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/gfram.git
cd gfram
```

### 2. Add Upstream Remote

```bash
git remote add upstream https://github.com/feruza-42h/gfram.git
git fetch upstream
```

### 3. Create a Branch

```bash
# For features:
git checkout -b feature/your-feature-name

# For bug fixes:
git checkout -b fix/issue-description

# For documentation:
git checkout -b docs/what-you-are-documenting
```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip and virtualenv
- Git

### Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import gfram; print(gfram.__version__)"
```

### Install Pre-commit Hooks (Recommended)

```bash
pip install pre-commit
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### IDE Setup

#### VS Code
```json
// .vscode/settings.json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"],
    "editor.formatOnSave": true
}
```

#### PyCharm
1. Settings → Tools → Black → Enable
2. Settings → Editor → Code Style → Set line length to 100
3. Enable "Optimize imports on save"

## Development Guidelines

### Code Style

We follow PEP 8 with some modifications:

- **Line length**: 100 characters maximum
- **Formatter**: Black
- **Import sorting**: isort
- **Type hints**: Required for all public functions

#### Example Code Style

```python
"""Module docstring describing the module purpose."""

from typing import List, Optional, Union

import numpy as np
import torch
from torch import nn


class GeometricFeature:
    """
    Compute geometric features from facial landmarks.
    
    This class extracts various geometric descriptors including
    distances, angles, and shape properties.
    
    Attributes:
        landmarks: Array of shape (478, 3) containing 3D coordinates.
        normalized: Whether landmarks are normalized.
    
    Example:
        >>> feature = GeometricFeature(landmarks)
        >>> distances = feature.compute_distances()
        >>> print(f"Computed {len(distances)} distances")
    """
    
    def __init__(
        self,
        landmarks: np.ndarray,
        normalize: bool = True,
        method: str = "standard"
    ) -> None:
        """
        Initialize GeometricFeature extractor.
        
        Args:
            landmarks: Facial landmarks array of shape (478, 3).
            normalize: Whether to normalize landmarks before processing.
            method: Normalization method, one of "standard", "robust", "minmax".
        
        Raises:
            ValueError: If landmarks shape is invalid.
            TypeError: If landmarks is not a numpy array.
        """
        if not isinstance(landmarks, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(landmarks)}")
        
        if landmarks.shape != (478, 3):
            raise ValueError(f"Expected shape (478, 3), got {landmarks.shape}")
        
        self.landmarks = landmarks
        self.normalized = False
        
        if normalize:
            self._normalize(method)
    
    def compute_distances(self, indices: Optional[List[tuple]] = None) -> np.ndarray:
        """
        Compute Euclidean distances between landmark pairs.
        
        Args:
            indices: List of (i, j) tuples specifying landmark pairs.
                    If None, uses default set of 45 pairs.
        
        Returns:
            Array of computed distances.
        
        Example:
            >>> distances = feature.compute_distances([(0, 1), (2, 3)])
            >>> print(distances)  # [0.123, 0.456]
        """
        if indices is None:
            indices = self._get_default_distance_pairs()
        
        distances = []
        for i, j in indices:
            d = np.linalg.norm(self.landmarks[i] - self.landmarks[j])
            distances.append(d)
        
        return np.array(distances)
    
    def _normalize(self, method: str) -> None:
        """Normalize landmarks using specified method."""
        # Implementation...
        self.normalized = True
    
    def _get_default_distance_pairs(self) -> List[tuple]:
        """Return default list of landmark pairs for distance computation."""
        # Implementation...
        return []
```

### Testing

We use pytest for testing. All new code must have tests.

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=gfram --cov-report=html --cov-report=term-missing

# Run specific test file
pytest tests/test_geometry.py

# Run specific test function
pytest tests/test_geometry.py::test_distance_computation

# Run tests with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "distance"
```

#### Writing Tests

```python
"""Tests for geometry module."""

import numpy as np
import pytest

from gfram.geometry import GeometricFeature


class TestGeometricFeature:
    """Test suite for GeometricFeature class."""
    
    @pytest.fixture
    def sample_landmarks(self):
        """Create sample landmarks for testing."""
        return np.random.randn(478, 3)
    
    @pytest.fixture
    def feature_extractor(self, sample_landmarks):
        """Create GeometricFeature instance."""
        return GeometricFeature(sample_landmarks)
    
    def test_initialization(self, sample_landmarks):
        """Test that GeometricFeature initializes correctly."""
        feature = GeometricFeature(sample_landmarks)
        assert feature.normalized is True
        assert feature.landmarks.shape == (478, 3)
    
    def test_invalid_shape_raises_error(self):
        """Test that invalid landmark shape raises ValueError."""
        invalid_landmarks = np.random.randn(68, 2)
        with pytest.raises(ValueError, match="Expected shape"):
            GeometricFeature(invalid_landmarks)
    
    def test_compute_distances_default(self, feature_extractor):
        """Test distance computation with default pairs."""
        distances = feature_extractor.compute_distances()
        assert len(distances) == 45
        assert all(d >= 0 for d in distances)
    
    def test_compute_distances_custom_pairs(self, feature_extractor):
        """Test distance computation with custom pairs."""
        pairs = [(0, 1), (10, 20), (100, 200)]
        distances = feature_extractor.compute_distances(indices=pairs)
        assert len(distances) == 3
    
    @pytest.mark.parametrize("method", ["standard", "robust", "minmax"])
    def test_normalization_methods(self, sample_landmarks, method):
        """Test different normalization methods."""
        feature = GeometricFeature(sample_landmarks, method=method)
        assert feature.normalized is True
```

### Code Formatting

```bash
# Format code with Black
black gfram/ tests/

# Sort imports with isort
isort gfram/ tests/

# Check with flake8
flake8 gfram/ tests/

# Type checking with mypy
mypy gfram/

# Run all checks
make lint  # If Makefile is available
```

### Documentation

We use Google-style docstrings. All public APIs must be documented.

```python
def compute_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    method: str = "cosine"
) -> float:
    """
    Compute similarity between two embeddings.
    
    This function supports multiple similarity metrics including
    cosine similarity, Euclidean distance, and dot product.
    
    Args:
        embedding1: First embedding vector of shape (512,).
        embedding2: Second embedding vector of shape (512,).
        method: Similarity method. Options are:
            - "cosine": Cosine similarity (default)
            - "euclidean": Inverse Euclidean distance
            - "dot": Dot product
    
    Returns:
        Similarity score in range [0, 1] for cosine and euclidean,
        unbounded for dot product.
    
    Raises:
        ValueError: If embeddings have different shapes.
        KeyError: If method is not recognized.
    
    Example:
        >>> emb1 = np.random.randn(512)
        >>> emb2 = np.random.randn(512)
        >>> sim = compute_similarity(emb1, emb2)
        >>> print(f"Similarity: {sim:.4f}")
    
    Note:
        For face verification, a threshold of 0.68 is recommended
        for cosine similarity.
    
    See Also:
        - :func:`compute_distance`: For distance computation
        - :class:`HybridMatcher`: For full matching pipeline
    """
    ...
```

## Making Changes

### 1. Sync with Upstream

```bash
git fetch upstream
git rebase upstream/main
```

### 2. Make Your Changes

- Write clean, documented code
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run tests
pytest

# Check code style
black --check gfram/
flake8 gfram/
mypy gfram/
```

### 4. Commit Your Changes

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Feature
git commit -m "feat: add support for GPU acceleration"

# Bug fix
git commit -m "fix: resolve memory leak in landmark detection"

# Documentation
git commit -m "docs: update installation instructions"

# Performance
git commit -m "perf: optimize distance computation by 2x"

# Refactoring
git commit -m "refactor: simplify feature extraction pipeline"

# Tests
git commit -m "test: add tests for edge cases in normalization"

# Chore (maintenance)
git commit -m "chore: update dependencies"
```

**Commit Message Format:**
```
<type>(<scope>): <short description>

<body - optional>

<footer - optional>
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Pull Request Process

### Before Submitting

Ensure your PR:

- [ ] All tests pass (`pytest`)
- [ ] Code is formatted (`black`, `isort`)
- [ ] No linting errors (`flake8`)
- [ ] Type hints are correct (`mypy`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (for significant changes)
- [ ] Commits follow conventional format

### PR Description Template

```markdown
## Description

Brief description of what this PR does.

## Motivation

Why is this change needed?

## Type of Change

- [ ] 🐛 Bug fix (non-breaking change fixing an issue)
- [ ] ✨ New feature (non-breaking change adding functionality)
- [ ] 💥 Breaking change (fix or feature causing existing functionality to change)
- [ ] 📚 Documentation update
- [ ] 🔧 Configuration/build change
- [ ] ♻️ Refactoring (no functional changes)

## How Has This Been Tested?

Describe tests you ran to verify your changes.

```bash
pytest tests/test_your_feature.py -v
```

## Screenshots (if applicable)

Add screenshots for UI changes.

## Checklist

- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code where necessary
- [ ] I have updated the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix/feature works
- [ ] New and existing tests pass locally
```

### Review Process

1. Automated checks run (CI)
2. Maintainer reviews code
3. Address feedback if any
4. Maintainer approves and merges

## Reporting Issues

### Bug Reports

Please include:

```markdown
## Bug Description
Clear description of the bug.

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.5]
- GFRAM version: [e.g., 2.0.0]
- PyTorch version: [e.g., 2.0.1]

## Steps to Reproduce
1. Step one
2. Step two
3. ...

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Error Message/Traceback
```
Paste error message here
```

## Additional Context
Any other relevant information.
```

### Feature Requests

Please include:

```markdown
## Feature Description
Clear description of the feature.

## Use Case
Why do you need this feature?

## Proposed Solution
How do you think it should work?

## Alternatives Considered
Other solutions you've thought about.

## Additional Context
Any other relevant information.
```

## Code of Conduct

### Our Pledge

We are committed to making participation in this project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity, experience level, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behaviors include:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behaviors include:**
- Trolling, insulting/derogatory comments, personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at feruzaortiqova42@gmail.com. All complaints will be reviewed and investigated promptly and fairly.

## Questions?

- 💬 Open a [Discussion](https://github.com/feruza-42h/gfram/discussions)
- 🐛 Open an [Issue](https://github.com/feruza-42h/gfram/issues)
- 📧 Email: feruzaortiqova42@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

**Thank you for contributing to GFRAM! 🙏**

Your contributions help make face recognition more accessible, efficient, and privacy-preserving.