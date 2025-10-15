"""
Setup script for the Milvus_Ops package.
"""

from setuptools import setup, find_packages

setup(
    name="milvus_ops",
    version="0.1.0",
    description="Milvus Vector Database Operations Package",
    author="Shubham Singh",
    author_email="shubh2014shiv@gmail.com",
    packages=find_packages(),
    install_requires=[
        "pymilvus>=2.3.0",
        "numpy>=1.20.0",
        "pydantic>=2.0.0,<3.0.0",
        "pydantic-settings>=2.0.0",
        "pydantic-yaml>=1.1.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "pandas>=1.3.0",
        "tenacity>=8.0.0",  # For retry logic
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
        "monitoring": [
            "prometheus-client>=0.16.0",
            "grafana-client>=2.2.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
)