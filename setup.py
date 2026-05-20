"""Setup configuration for the NLP Pipeline package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
long_description = ""
try:
    long_description = Path("README.md").read_text(encoding="utf-8")
except FileNotFoundError:
    pass

# Read requirements
requirements = []
try:
    with open("requirements.txt", "r") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]
except FileNotFoundError:
    pass

setup(
    name="nlp-pipeline-information-extraction",
    version="1.0.0",
    author="Venkateswar",
    description=(
        "Production-grade transformer-based NLP pipeline for "
        "structured information extraction from documents."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Venkateswar8703/NLP-Pipeline-for-Information-Extraction",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0",
            "flake8>=6.0",
            "isort>=5.12",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
    entry_points={
        "console_scripts": [
            "nlp-pipeline=scripts.run_pipeline:main",
        ],
    },
)
