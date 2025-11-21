# Contributing to GFRAM

Thank you for your interest in contributing to GFRAM! 🎉

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
git clone https://github.com/feruza-42h/gfram.git
cd gfram
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

## Development Guidelines

### Code Style

- Follow PEP 8 style guide
- Use type hints where possible
- Add docstrings to all public functions and classes
- Maximum line length: 100 characters

```python
def example_function(param: int) -> str:
    """
    Brief description.
    
    Args:
        param: Description of parameter.
    
    Returns:
        Description of return value.
    """
    return str(param)
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gfram --cov-report=html

# Run specific test
pytest tests/test_geometry.py
```

### Code Formatting

```bash
# Format code with black
black gfram/

# Check with flake8
flake8 gfram/

# Type checking with mypy
mypy gfram/
```

## Making Changes

### 1. Write Tests

- Add tests for new features
- Ensure all tests pass
- Aim for >80% code coverage

### 2. Update Documentation

- Update docstrings
- Add examples if needed
- Update README.md if necessary

### 3. Commit Changes

```bash
git add .
git commit -m "feat: add new feature"
# or
git commit -m "fix: resolve issue #123"
```

**Commit Message Format**:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `test:` - Adding tests
- `chore:` - Maintenance tasks

### 4. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Pull Request Guidelines

### Before Submitting

- [ ] All tests pass
- [ ] Code is formatted (black)
- [ ] No linting errors (flake8)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (if applicable)

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe how you tested your changes

## Checklist
- [ ] Tests pass
- [ ] Code formatted
- [ ] Documentation updated
```

## Reporting Issues

### Bug Reports

Include:
- Python version
- OS and version
- Steps to reproduce
- Expected vs actual behavior
- Error messages/traceback

### Feature Requests

Include:
- Use case
- Proposed solution
- Alternative solutions considered

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Accept constructive criticism
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing private information
- Unprofessional conduct

## Questions?

- Open an issue for discussion
- Check existing issues and PRs
- Read the documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to GFRAM! 🙏**