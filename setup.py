#!/usr/bin/env python3
"""Setup script for Shvayambhu LLM."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="shvayambhu",
    version="0.1.0",
    author="Shvayambhu Team",
    author_email="team@shvayambhu.ai",
    description="Self-evolving offline LLM for Apple Silicon",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shvayambhu/shvayambhu",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: MacOS :: MacOS X",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.4.0", "black==23.7.0", "isort>=5.12.0", "flake8>=6.0.0", "mypy>=1.4.0"],
        "docs": ["sphinx>=7.0.0", "sphinx-rtd-theme>=1.3.0", "mkdocs>=1.5.0"],
    },
    entry_points={
        "console_scripts": [
            "shvayambhu=shvayambhu.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)