"""
GFRAM - Geometric Face Recognition and Matching
Setup file for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
req_file = this_directory / "requirements.txt"
if req_file.exists():
    with open(req_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="gfram",
    version="3.2.0",
    author="Ortiqova F.S.",
    author_email="feruzaortiqova42@gmail.com",
    description="Professional geometric face recognition library with AI-powered matching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/feruza-42h/gfram",
    project_urls={
        "Bug Tracker": "https://github.com/feruza-42h/gfram/issues",
        "Documentation": "https://gfram.readthedocs.io",
        "Source Code": "https://github.com/feruza-42h/gfram",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    # extras (dev) are declared in pyproject.toml
    include_package_data=True,
    zip_safe=False,
)