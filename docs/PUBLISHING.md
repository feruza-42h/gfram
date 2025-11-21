# Publishing GFRAM to PyPI

This guide explains how to publish the GFRAM library to PyPI.

## Prerequisites

1. Install build tools:
```bash
pip install build twine
```

2. Create accounts:
   - PyPI: https://pypi.org/account/register/
   - TestPyPI (for testing): https://test.pypi.org/account/register/

## Build the Package

1. Clean previous builds:
```bash
cd /path/to/gfram-v2
rm -rf dist/ build/ *.egg-info
```

2. Build the package:
```bash
python -m build
```

This creates:
- `dist/gfram-2.0.1-py3-none-any.whl` (wheel)
- `dist/gfram-2.0.1.tar.gz` (source distribution)

## Test on TestPyPI First

1. Upload to TestPyPI:
```bash
twine upload --repository testpypi dist/*
```

2. Install from TestPyPI to test:
```bash
pip install --index-url https://test.pypi.org/simple/ gfram
```

3. Test the installation:
```python
import gfram
print(gfram.__version__)
```

## Publish to PyPI

1. Upload to PyPI:
```bash
twine upload dist/*
```

2. Verify on PyPI:
   - Visit: https://pypi.org/project/gfram/
   - Check that all information is correct

3. Install and test:
```bash
pip install gfram
```

## Updating the Package

When releasing a new version:

1. Update version in `gfram/version.py`:
```python
__version__ = "2.0.1"
```

2. Update `pyproject.toml`:
```toml
version = "2.0.1"
```

3. Create a git tag:
```bash
git tag v2.0.1
git push origin v2.0.1
```

4. Rebuild and upload:
```bash
rm -rf dist/
python -m build
twine upload dist/*
```

## Configuration Files

### .pypirc (Optional)

Create `~/.pypirc` for easier uploading:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TOKEN_HERE
```

## Checklist Before Publishing

- [ ] All tests pass
- [ ] Documentation is complete and up-to-date
- [ ] README.md is informative
- [ ] LICENSE file is included
- [ ] Version number is updated
- [ ] CHANGELOG.md is updated (if exists)
- [ ] Examples work correctly
- [ ] Dependencies are correctly specified
- [ ] Package builds without errors
- [ ] Tested on TestPyPI

## Common Issues

### Issue: ModuleNotFoundError during build
**Solution**: Ensure all dependencies are listed in `pyproject.toml`

### Issue: Package too large
**Solution**: 
- Check MANIFEST.in
- Exclude unnecessary files
- Consider using `.gitignore` patterns

### Issue: Import errors after installation
**Solution**:
- Check `__init__.py` files
- Verify package structure
- Test in clean virtual environment

## Post-Publication

1. Announce the release:
   - GitHub Releases
   - Twitter/Social Media
   - Research communities
   - ML forums

2. Monitor:
   - PyPI download statistics
   - GitHub issues
   - User feedback

3. Update documentation:
   - Installation instructions
   - API changes
   - Migration guides

## Resources

- PyPI Documentation: https://packaging.python.org/
- PyPI Project Page: https://pypi.org/project/gfram/
- TestPyPI: https://test.pypi.org/
- Packaging Tutorial: https://packaging.python.org/tutorials/packaging-projects/

## Support

For issues with publishing, contact:
- PyPI Support: https://pypi.org/help/
- GitHub Issues: https://github.com/feruza-42h/gfram/issues